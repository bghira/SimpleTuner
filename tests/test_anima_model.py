import json
import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
