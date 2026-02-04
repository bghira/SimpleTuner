import os
import tempfile
import unittest

from simpletuner.simpletuner_sdk.server.services.field_service import FieldService


class FieldServiceResumeCheckpointTests(unittest.TestCase):
    def test_resume_from_checkpoint_options_include_local_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "checkpoint-100"))
            os.makedirs(os.path.join(tmpdir, "checkpoint-200"))
            os.makedirs(os.path.join(tmpdir, "not-a-checkpoint"))

            service = FieldService()
            fields, _sections = service.build_template_tab(
                "basic",
                {"output_dir": tmpdir},
                raw_config={"output_dir": tmpdir},
            )

            resume_field = next(field for field in fields if field["name"] == "resume_from_checkpoint")
            values = [opt["value"] for opt in resume_field.get("options", [])]

            self.assertIn("latest", values)
            self.assertIn("", values)
            self.assertIn("checkpoint-200", values)
            self.assertIn("checkpoint-100", values)


if __name__ == "__main__":
    unittest.main()
