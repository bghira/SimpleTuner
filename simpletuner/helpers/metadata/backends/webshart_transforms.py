import json
import random
from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.multiaspect.image import ImageGeometry


@dataclass(frozen=True)
class PreparationGeometry:
    config: dict
    sizing: ImageGeometry
    seed: int
    model: object

    def rng_for(self, ref):
        if self.config.get("crop") and (
            self.config.get("crop_style", "random") in {"random", "face"} or self.config.get("crop_aspect") == "random"
        ):
            return random.Random(f"{self.seed}:{ref}")
        return None

    def prepare(self, dataset_id, ref, metadata):
        return TrainingSample(
            image=None,
            data_backend_id=dataset_id,
            image_metadata=metadata,
            image_path=ref,
            model=self.model,
            data_backend_config=self.config,
            geometry=self.sizing,
            rng=self.rng_for(ref),
        ).prepare()

    def cache_settings(self):
        settings = (
            "resolution",
            "resolution_type",
            "crop",
            "crop_style",
            "crop_aspect",
            "crop_aspect_buckets",
            "target_downsample_size",
            "maximum_image_size",
            "max_upscale_threshold",
        )
        return {
            "version": 1,
            "config": {key: self.config[key] for key in settings if key in self.config},
            "alignment": self.sizing.alignment,
            "rounding": self.sizing.rounding,
            "seed": self.seed,
            "canvas_limit": getattr(self.model, "MAXIMUM_CANVAS_SIZE", None),
        }

    @classmethod
    def from_cache_settings(cls, settings):
        if settings["version"] != 1:
            raise ValueError("Unsupported Webshart transform recipe version.")
        return cls(
            settings["config"],
            ImageGeometry(settings["alignment"], settings["rounding"]),
            settings["seed"],
            SimpleNamespace(MAXIMUM_CANVAS_SIZE=settings["canvas_limit"]),
        )


def prepared_fields(sample):
    return {
        "aspect_ratio": float(sample.aspect_ratio),
        "intermediary_size": sample.intermediary_size,
        "crop_coordinates": sample.crop_coordinates,
        "target_size": sample.target_size,
    }


class WebshartTransforms:
    """Bind a native transform cache to a pinned Parquet view and its recipe."""

    _MAX_BATCH_BYTES = 1 << 20
    _PROJECTED_FIELDS = ("width", "height", "offset", "size", "json_path", "json_metadata")

    def __init__(self, source, settings, shard_revision):
        self.source = source
        self.recipe = PreparationGeometry.from_cache_settings(settings)
        self.shard_revision = shard_revision
        cache_type = getattr(source.webshart, "TransformCache", None)
        if cache_type is None:
            raise ImportError("This prepared Webshart view requires native Webshart TransformCache support.")
        scope = json.dumps(
            [settings, source.columns, source.metadata, source.aws_endpoint_url], sort_keys=True, separators=(",", ":")
        )
        directory = Path(source.cache_dir) / ".webshart-transforms"
        self.cache = cache_type(directory, sha256(scope.encode()).hexdigest(), max_batch_bytes=self._MAX_BATCH_BYTES)

    def _shard_key(self, index):
        revision = self.shard_revision(index)
        if revision is None:
            raise ValueError("The prepared Webshart view has no content revision for this shard.")
        shard = self.source.dataset.get_shard_info(index)
        # Seeded crops use the complete sample reference, including the shard position.
        return json.dumps([self.source.source, shard["path"], revision, index], separators=(",", ":"))

    @staticmethod
    def _provenance(ref, row):
        return {
            "shard_idx": ref.shard_idx,
            "sample_idx": ref.sample_idx,
            "filename": ref.filename,
            "offset": row.get("offset"),
            "size": row.get("size"),
            "json_path": row.get("json_path"),
        }

    def enqueue(self, records):
        groups = defaultdict(list)
        for ref, record in records.items():
            location = self.source.parse_sample_id(ref)
            payload = {key: value for key, value in record.items() if key != "webshart"}
            provenance = record["webshart"]
            payload["webshart"] = {
                key: provenance[key] for key in ("offset", "size", "json_path") if provenance.get(key) is not None
            }
            groups[location.shard_idx].append((location.sample_idx, json.dumps(payload, separators=(",", ":")).encode()))
        for shard, entries in groups.items():
            key = self._shard_key(shard)
            remaining = self._MAX_BATCH_BYTES - len(key.encode())
            batch, used = [], 0
            for row, value in entries:
                size = len(value) + 64
                if size > remaining:
                    # A large source metadata value can still be returned on a cache miss.
                    continue
                if used + size > remaining:
                    self.cache.enqueue(key, batch)
                    batch, used = [], 0
                batch.append((row, value))
                used += size
            if batch:
                self.cache.enqueue(key, batch)

    def read_batch(self, refs):
        locations = [self.source.parse_sample_id(ref) for ref in refs]
        keys = {shard: self._shard_key(shard) for shard in dict.fromkeys(location.shard_idx for location in locations)}
        payloads = self.cache.read_batch([(keys[location.shard_idx], location.sample_idx) for location in locations])
        results = []
        missing = defaultdict(dict)
        for index, (ref, location, payload) in enumerate(zip(refs, locations, payloads, strict=True)):
            if payload is None:
                missing[location.shard_idx][ref] = location
                results.append(None)
            else:
                record = json.loads(payload)
                record["webshart"] = self._provenance(location, record["webshart"])
                results.append(record)
        if missing:
            computed = {}
            for shard, group in missing.items():
                rows = self.source.dataset.read_metadata_rows(
                    shard,
                    [location.sample_idx for location in group.values()],
                    fields=self._PROJECTED_FIELDS,
                    parallel_reads=self.source.parallel_downloads,
                )
                for (ref, location), row in zip(group.items(), rows, strict=True):
                    record = {
                        "original_size": [int(row["width"]), int(row["height"])],
                        "webshart": self._provenance(location, row),
                        "prepared_geometry": True,
                    }
                    if row.get("json_metadata"):
                        record["json_metadata"] = row["json_metadata"]
                    if "json_path" in row:
                        record["json_path"] = row["json_path"]
                    fields = prepared_fields(self.recipe.prepare(self.source.id, ref, record))
                    record.update({key: list(value) if isinstance(value, tuple) else value for key, value in fields.items()})
                    computed[ref] = record
            self.enqueue(computed)
            results = [computed[ref] if result is None else result for ref, result in zip(refs, results, strict=True)]
        return results

    def flush(self):
        self.cache.flush()

    def close(self):
        self.cache.close()
