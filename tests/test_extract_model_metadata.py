import types
import unittest

from scripts.extract_model_metadata import extract_metadata_from_module


class ExtractModelMetadataTestCase(unittest.TestCase):
    def test_ignores_imported_model_classes(self) -> None:
        module = types.ModuleType("simpletuner.helpers.models.boogu_image.model")

        class ImportedFlux:
            NAME = "Flux.1"

            @classmethod
            def get_flavour_choices(cls):
                return ["dev"]

        class LocalBooguImage:
            NAME = "Boogu-Image"
            PREDICTION_TYPE = "flow_matching"

            @classmethod
            def get_flavour_choices(cls):
                return ["v0.1-base", "v0.1-turbo"]

        ImportedFlux.__module__ = "simpletuner.helpers.models.flux.model"
        LocalBooguImage.__module__ = module.__name__
        module.Flux = ImportedFlux
        module.BooguImage = LocalBooguImage

        result = extract_metadata_from_module(
            module,
            "simpletuner.helpers.models.boogu_image.model",
            "boogu_image",
        )

        self.assertEqual(result["class_name"], "BooguImage")
        self.assertEqual(result["name"], "Boogu-Image")
        self.assertEqual(result["flavour_choices"], ["v0.1-base", "v0.1-turbo"])


if __name__ == "__main__":
    unittest.main()
