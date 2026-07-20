import unittest

from simpletuner.simpletuner_sdk.server.data.dataset_blueprints import find_blueprint, get_blueprint_lookup


class TestMemoryDatasetBlueprints(unittest.TestCase):
    def test_memory_blueprints_only_cover_cache_dataset_types(self):
        lookup = get_blueprint_lookup()

        memory_types = {dataset_type for backend_type, dataset_type in lookup if backend_type == "memory"}
        self.assertEqual(memory_types, {"text_embeds", "image_embeds"})

    def test_memory_blueprint_exposes_tmpfs_settings(self):
        for dataset_type in ("text_embeds", "image_embeds"):
            blueprint = find_blueprint("memory", dataset_type)

            self.assertIsNotNone(blueprint)
            self.assertEqual(blueprint.defaults["type"], "memory")
            fields = {field.id: field for field in blueprint.fields}
            self.assertIn("cache_dir", fields)
            self.assertIn("memory_filesystem_path", fields)
            self.assertIn("memory_filesystem_size", fields)
            self.assertIn("memory_filesystem_sudo", fields)
            self.assertFalse(fields["memory_filesystem_sudo"].defaultValue)


if __name__ == "__main__":
    unittest.main()
