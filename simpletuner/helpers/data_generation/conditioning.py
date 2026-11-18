import logging
import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import batched
from types import SimpleNamespace

from PIL import Image

from simpletuner.helpers.data_generation.sample_generator import SampleGenerator

logger = logging.getLogger(__name__)

_IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}


def _transform_batch(generator, device, targets, records):
    images = generator.transform_batch(
        images=[record["image"] for record in records],
        source_paths=[record["image_path"] for record in records],
        metadata_list=records,
        accelerator=SimpleNamespace(device=device),
    )
    contents = []
    for path, image in zip(targets, images, strict=True):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Conditioning generator returned {type(image).__name__} for {path}; expected a PIL image.")
        image_format = _IMAGE_FORMATS.get(os.path.splitext(path)[1].lower(), "PNG")
        contents.append(generator.image_to_bytes(image, format=image_format))
    return contents


class DataGenerator:
    """Generate a preparation partition using bounded reads and transforms."""

    def __init__(self, source, target, accelerator, *, read_batch_size=24, transform_batch_size=4, num_workers=None):
        self.source = source
        self.target = target
        self.accelerator = accelerator
        self.read_batch_size = read_batch_size
        self.transform_batch_size = transform_batch_size
        self.sample_generator = SampleGenerator.from_backend(target.config)
        self.num_workers = num_workers if num_workers is not None else max(1, min(multiprocessing.cpu_count() - 2, 8))

    def _pending_batches(self):
        for _, refs in self.source.iter_preparation_batches(
            worker=self.accelerator.process_index,
            workers=self.accelerator.num_processes,
            batch_size=self.read_batch_size,
        ):
            paths = [self.target.conditioning_ref(self.source.pairing_key(ref)) for ref in refs]
            present = self.target.data_backend.bulk_exists(paths)
            pending = [(ref, path) for ref, path, exists in zip(refs, paths, present, strict=True) if not exists]
            if pending:
                records = self.source.read_samples([ref for ref, _ in pending], fields=("image", "metadata"))
                for batch in batched(zip(pending, records, strict=True), self.transform_batch_size):
                    yield [path for (_, path), _ in batch], [record for _, record in batch]

    def generate_dataset(self):
        transform = partial(_transform_batch, self.sample_generator, self.accelerator.device)
        if getattr(self.sample_generator, "requires_gpu", False) or not self.num_workers:
            for paths, records in self._pending_batches():
                self.target.data_backend.write_batch(paths, transform(paths, records))
        else:
            with ProcessPoolExecutor(
                max_workers=self.num_workers, mp_context=multiprocessing.get_context("spawn")
            ) as executor:
                pending = deque()
                for paths, records in self._pending_batches():
                    pending.append((paths, executor.submit(transform, paths, records)))
                    if len(pending) >= self.num_workers:
                        self._write_completed(pending)
                while pending:
                    self._write_completed(pending)
        self.accelerator.wait_for_everyone()
        logger.info("Conditioning generation complete for %s", self.target.id)

    def _write_completed(self, pending):
        paths, result = pending.popleft()
        self.target.data_backend.write_batch(paths, result.result())
