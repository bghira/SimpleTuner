import json
import unittest
from pathlib import Path
from unittest.mock import patch


class TestAnimaModel(unittest.TestCase):
    def test_model_import(self):
        from simpletuner.helpers.models.anima.model import Anima

        self.assertIsNotNone(Anima)
        self.assertEqual(Anima.NAME, "Anima")
        self.assertEqual(Anima.DEFAULT_MODEL_FLAVOUR, "preview")

    def test_transformer_import(self):
        from simpletuner.helpers.models.anima.transformer import AnimaTransformerModel

        self.assertIsNotNone(AnimaTransformerModel)
        self.assertTrue(hasattr(AnimaTransformerModel, "from_pretrained"))
        self.assertTrue(hasattr(AnimaTransformerModel, "from_single_file"))

    def test_pipeline_import(self):
        from simpletuner.helpers.models.anima.pipeline import AnimaPipeline

        self.assertIsNotNone(AnimaPipeline)
        self.assertTrue(hasattr(AnimaPipeline, "from_pretrained"))
        self.assertTrue(hasattr(AnimaPipeline, "from_single_file"))

    def test_vendored_headers_present(self):
        anima_dir = Path(__file__).parent.parent / "simpletuner/helpers/models/anima"
        for relative_path in ("transformer.py", "pipeline.py", "loading.py", "scheduler.py"):
            contents = (anima_dir / relative_path).read_text()
            self.assertTrue(contents.startswith("# Vendored from diffusers-anima: "))

    def test_metadata_contains_anima(self):
        metadata_path = Path(__file__).parent.parent / "simpletuner/helpers/models/model_metadata.json"
        with open(metadata_path) as handle:
            metadata = json.load(handle)

        self.assertIn("anima", metadata)
        self.assertEqual(metadata["anima"]["class_name"], "Anima")
        self.assertEqual(metadata["anima"]["module_path"], "simpletuner.helpers.models.anima.model")

    def test_tokenizer_loader_ignores_proxies(self):
        from simpletuner.helpers.models.anima.loading import AnimaLoaderOptions, load_tokenizer_from_source

        options = AnimaLoaderOptions(local_files_only=True, proxies={"https": "http://proxy"})
        with patch("simpletuner.helpers.models.anima.loading.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            load_tokenizer_from_source("repo::tokenizer", options=options)

        self.assertNotIn("proxies", mock_from_pretrained.call_args.kwargs)

    def test_lora_loader_ignores_proxies(self):
        from simpletuner.helpers.models.anima.lora_pipeline import _fetch_anima_lora_state_dict

        with (
            patch("simpletuner.helpers.models.anima.lora_pipeline.hf_hub_download") as mock_download,
            patch("simpletuner.helpers.models.anima.lora_pipeline.torch.load") as mock_torch_load,
        ):
            mock_download.return_value = "weights.bin"
            mock_torch_load.return_value = {"transformer.block.lora_A.weight": object()}
            _fetch_anima_lora_state_dict(
                "repo-id",
                weight_name="weights.bin",
                use_safetensors=False,
                local_files_only=True,
                cache_dir=None,
                force_download=False,
                proxies={"https": "http://proxy"},
                token=None,
                revision=None,
                subfolder=None,
                allow_pickle=True,
            )

        self.assertNotIn("proxies", mock_download.call_args.kwargs)

    def test_transformer_validation_guards(self):
        from simpletuner.helpers.models.anima.transformer import _AdapterAttention, _RotaryEmbedding

        with self.assertRaisesRegex(ValueError, "even"):
            _RotaryEmbedding(63)

        with self.assertRaisesRegex(ValueError, "divisible"):
            _AdapterAttention(query_dim=1025, context_dim=1024, heads=16)


if __name__ == "__main__":
    unittest.main()
