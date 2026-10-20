import multiprocessing
import shutil
import tempfile
import textwrap
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import torch

from simpletuner.helpers.training.crepa import CrepaRegularizer, UrepaRegularizer


def _load_dino_from_shared_cache(
    cache_dir, archive, start, ready, downloading, duplicate_download, release, count, results, kind
):
    torch.set_num_threads(1)
    torch.hub.set_dir(cache_dir)

    def download_archive(url, destination, progress):
        with count.get_lock():
            count.value += 1
            if count.value > 1:
                duplicate_download.set()
        downloading.set()
        if not release.wait(timeout=20):
            raise TimeoutError("Test did not release the archive download")
        shutil.copyfile(archive, destination)

    regularizer_cls = CrepaRegularizer if kind == "crepa" else UrepaRegularizer
    regularizer = regularizer_cls.__new__(regularizer_cls)
    regularizer.encoder = None
    regularizer.encoder_name = "dinov2_vits14" if kind == "crepa" else "dinov2_vitb14"
    regularizer.encoder_image_size = 28
    regularizer.device = torch.device("cpu")
    try:
        with (
            patch("torch.hub._parse_repo_info", return_value=("facebookresearch", "dinov2", "main")),
            patch("torch.hub._validate_not_a_forked_repo"),
            patch("torch.hub.download_url_to_file", side_effect=download_archive),
        ):
            ready.put(kind)
            if not start.wait(timeout=20):
                raise TimeoutError("Test did not start encoder loading")
            regularizer._load_encoder()
            encoder = regularizer.encoder
            results.put((kind, "ok", regularizer.encoder_dim, encoder.training, next(encoder.parameters()).requires_grad))
    except Exception as error:
        results.put((kind, "error", type(error).__name__, str(error)))
    finally:
        ready.close()
        ready.join_thread()
        results.close()
        results.join_thread()


class DinoCacheTests(unittest.TestCase):
    def test_processes_share_cold_and_warm_cache_across_encoder_variants(self):
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache = root / "hub"
            cache.mkdir()
            (cache / "trusted_list").write_text("facebookresearch_dinov2\n")
            archive = root / "dinov2.zip"
            hubconf = textwrap.dedent(
                """\
                import torch

                class Encoder(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

                    def forward(self, images):
                        assert images.dtype == self.weight.dtype
                        return {"x_norm_patchtokens": self.weight * torch.ones(images.shape[0], 4, 8)}

                    forward_features = forward

                def dinov2_vits14():
                    return Encoder()

                def dinov2_vitb14():
                    return Encoder()
                """
            )
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("dinov2-main/", "")
                zip_file.writestr("dinov2-main/hubconf.py", hubconf)

            for cache_state, expected_downloads in (("cold", 1), ("warm", 0)):
                with self.subTest(cache_state=cache_state):
                    start = context.Event()
                    downloading = context.Event()
                    duplicate_download = context.Event()
                    release = context.Event()
                    ready = context.Queue()
                    results = context.Queue()
                    count = context.Value("i", 0)
                    processes = [
                        context.Process(
                            target=_load_dino_from_shared_cache,
                            args=(
                                str(cache),
                                str(archive),
                                start,
                                ready,
                                downloading,
                                duplicate_download,
                                release,
                                count,
                                results,
                                kind,
                            ),
                        )
                        for kind in ("crepa", "urepa")
                    ]
                    try:
                        for process in processes:
                            process.start()
                        for _ in processes:
                            ready.get(timeout=30)
                        start.set()
                        if cache_state == "cold":
                            self.assertTrue(downloading.wait(timeout=20))
                            duplicate_download.wait(timeout=0.5)
                        release.set()
                        outcomes = [results.get(timeout=30) for _ in processes]
                        for process in processes:
                            process.join(timeout=10)
                            self.assertEqual(process.exitcode, 0)
                        self.assertEqual(count.value, expected_downloads)
                        self.assertEqual(
                            sorted(outcomes), [("crepa", "ok", 8, False, False), ("urepa", "ok", 8, False, False)]
                        )
                    finally:
                        start.set()
                        release.set()
                        for process in processes:
                            if process.pid is not None:
                                process.join(timeout=5)
                                if process.is_alive():
                                    process.terminate()
                                    process.join(timeout=5)
                        ready.close()
                        results.close()


class QwenRepaTests(unittest.TestCase):
    def test_dino_uses_patch_features_not_cls_embedding(self):
        class Dino(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def forward(self, images):
                raise AssertionError("Default DINO forward returns only the CLS embedding")

            def forward_features(self, images):
                return {"x_norm_patchtokens": self.weight * torch.ones(images.shape[0], 4, 8)}

        reg = CrepaRegularizer.__new__(CrepaRegularizer)
        reg.encoder_name = "dinov2_vitg14"
        reg.encoder = Dino()
        features = reg._forward_encoder(torch.zeros(2, 3, 28, 28))
        self.assertEqual(features.shape, (2, 4, 8))
        self.assertFalse(features.requires_grad)
