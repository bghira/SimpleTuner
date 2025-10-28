import unittest

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.distillation.requirements import (
    EMPTY_PROFILE,
    DistillerRequirementProfile,
    describe_requirement_groups,
    evaluate_requirement_profile,
    parse_distiller_requirement_profile,
    parse_requirement_matrix,
)


class TestRequirementParsing(unittest.TestCase):
    def test_parse_requirement_matrix_handles_nested_and_flat_entries(self):
        requirements = parse_requirement_matrix([["image", "video"], "caption"])

        self.assertEqual(len(requirements), 2)
        self.assertTrue(requirements[0].is_satisfied_by(DatasetType.IMAGE))
        self.assertTrue(requirements[0].is_satisfied_by(DatasetType.VIDEO))
        self.assertFalse(requirements[0].is_satisfied_by(DatasetType.CAPTION))
        self.assertTrue(requirements[1].is_satisfied_by(DatasetType.CAPTION))

    def test_parse_requirement_profile_defaults_to_empty(self):
        profile = parse_distiller_requirement_profile({})
        self.assertEqual(profile, EMPTY_PROFILE)

        profile = parse_distiller_requirement_profile({"is_data_generator": True})
        self.assertIsInstance(profile, DistillerRequirementProfile)
        self.assertTrue(profile.is_data_generator)
        self.assertEqual(profile.requirements, ())

    def test_evaluate_requirement_profile_flags_missing_groups(self):
        profile = parse_distiller_requirement_profile({"data_requirements": ["caption"]})
        result = evaluate_requirement_profile(profile, [{"dataset_type": "image"}])
        self.assertFalse(result.fulfilled)
        self.assertEqual(len(result.missing_requirements), 1)
        self.assertIn("caption", describe_requirement_groups(result.missing_requirements))


class DummyDistiller(DistillationBase):
    """Minimal stub for registry tests."""

    def __init__(self):
        pass


class TestRegistryProfiles(unittest.TestCase):
    def setUp(self):
        self._registry_snapshot = (
            DistillationRegistry._registry.copy(),
            DistillationRegistry._metadata.copy(),
            DistillationRegistry._requirement_profiles.copy(),
        )

    def tearDown(self):
        DistillationRegistry._registry = self._registry_snapshot[0]
        DistillationRegistry._metadata = self._registry_snapshot[1]
        DistillationRegistry._requirement_profiles = self._registry_snapshot[2]

    def test_register_captures_requirement_profile(self):
        DistillationRegistry.register(
            "dummy",
            DummyDistiller,
            data_requirements=[["image", "video"], "caption"],
            is_data_generator=True,
            requirement_notes="Generates pixels once captions provided.",
        )

        profile = DistillationRegistry.get_requirement_profile("dummy")
        self.assertTrue(profile.is_data_generator)
        self.assertEqual(len(profile.requirements), 2)
        self.assertTrue(profile.requires_dataset_type(DatasetType.IMAGE))

        metadata = DistillationRegistry.get_metadata_with_requirements("dummy")
        self.assertIn("requirement_profile", metadata)
        self.assertEqual(metadata["requirement_profile"], profile)


if __name__ == "__main__":
    unittest.main()
