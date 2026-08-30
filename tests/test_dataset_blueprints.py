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


class TestTrainBatchSizeBlueprints(unittest.TestCase):
    def test_train_batch_size_only_applies_to_independently_sampled_datasets(self):
        lookup = get_blueprint_lookup()
        supported_types = {"image", "video", "audio", "caption"}

        for (_backend_type, dataset_type), blueprint in lookup.items():
            fields = {field.id for field in blueprint.fields}
            self.assertEqual(
                "train_batch_size" in fields,
                dataset_type in supported_types,
                f"Unexpected train_batch_size eligibility for {blueprint.backendType}/{dataset_type}",
            )

    def test_local_caption_blueprint_exposes_train_batch_size(self):
        blueprint = find_blueprint("local", "caption")

        self.assertIsNotNone(blueprint)
        self.assertIn("train_batch_size", {field.id for field in blueprint.fields})


class TestAudioDatasetBlueprints(unittest.TestCase):
    def test_local_audio_blueprint_exposes_data_transforms(self):
        blueprint = find_blueprint("local", "audio")

        self.assertIsNotNone(blueprint)
        fields = {field.id: field for field in blueprint.fields}
        self.assertIn("data_transforms", fields)
        self.assertEqual(fields["data_transforms"].type, "textarea")
        self.assertIn("identity_transfer", fields["data_transforms"].placeholder)


if __name__ == "__main__":
    unittest.main()
