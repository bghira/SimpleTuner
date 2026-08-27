import unittest

from simpletuner.helpers.acceleration.backends import AccelerationBackend
from simpletuner.helpers.utils import offloading
from simpletuner.simpletuner_sdk.server.services.field_registry.sections.training import register_training_fields


class _Registry:
    def __init__(self):
        self.fields = []

    def _add_field(self, field):
        self.fields.append(field)


class GroupOffloadRemovalTests(unittest.TestCase):
    def test_acceleration_backend_is_not_exposed(self):
        self.assertNotIn("GROUP_OFFLOAD", AccelerationBackend.__members__)

    def test_diffusers_group_offload_helper_is_not_exposed(self):
        self.assertFalse(hasattr(offloading, "enable_group_offload_on_components"))

    def test_training_registry_has_no_group_offload_fields(self):
        registry = _Registry()
        register_training_fields(registry)

        field_names = {field.name for field in registry.fields}
        self.assertFalse(any(name.startswith("group_offload") for name in field_names))
        self.assertNotIn("enable_group_offload", field_names)


if __name__ == "__main__":
    unittest.main()
