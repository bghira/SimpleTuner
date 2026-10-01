import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.model import QwenImage


class QwenTextureVAETests(unittest.TestCase):
    def make_model(self, **overrides):
        config = dict(
            model_family="qwen_image",
            model_flavour="v2.1",
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            revision="base-model-revision",
            variant="bf16",
            vae_enable_tiling=True,
            vae_enable_slicing=True,
        )
        config.update(overrides)
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(**config)
        model.setup_model_flavour()
        return model

    def test_default_and_custom_transformer_use_texture_fix(self):
        for overrides in (
            {},
            {"model_flavour": None},
            {"pretrained_model_name_or_path": "custom/model", "vae_path": "custom/model"},
        ):
            with self.subTest(overrides=overrides):
                model = self.make_model(**overrides)
                self.assertEqual(model.config.vae_path, QwenImage.TEXTURE_FIX_VAE_REPO)
                self.assertEqual(model.config.pretrained_vae_model_name_or_path, QwenImage.TEXTURE_FIX_VAE_REPO)
                self.assertEqual(
                    model._get_vae_load_kwargs(),
                    {
                        "pretrained_model_name_or_path": QwenImage.TEXTURE_FIX_VAE_REPO,
                        "subfolder": None,
                        "revision": QwenImage.TEXTURE_FIX_VAE_REVISION,
                        "force_upcast": False,
                        "variant": None,
                    },
                )
                self.assertEqual(model.config.revision, "base-model-revision")
                self.assertEqual(model.config.variant, "bf16")
                model.setup_model_flavour()
                self.assertEqual(model.config.vae_path, QwenImage.TEXTURE_FIX_VAE_REPO)

    def test_explicit_vae_including_original_is_preserved(self):
        for path in ("custom/vae", "Qwen/Qwen-Image-2.1"):
            model = self.make_model(pretrained_vae_model_name_or_path=path, vae_path=path)
            self.assertEqual(model.config.vae_path, path)
            self.assertEqual(model._get_vae_load_kwargs()["revision"], "base-model-revision")
        model = self.make_model(vae_path="local-vae")
        self.assertEqual(model.config.vae_path, "local-vae")

    def test_legacy_flavours_keep_bundled_vae(self):
        for flavour in ("v1.0", "v2.0", "edit-v1", "edit-v2", "edit-v3"):
            with self.subTest(flavour=flavour):
                model = self.make_model(model_flavour=flavour)
                self.assertEqual(model.config.vae_path, model.config.pretrained_model_name_or_path)
                self.assertEqual(model._get_vae_load_kwargs()["subfolder"], "vae")
                self.assertEqual(model._get_vae_load_kwargs()["variant"], "bf16")

    def test_shared_loader_loads_pinned_root_vae_and_reloads(self):
        model = self.make_model()
        model.AUTOENCODER_CLASS = AutoencoderKLQwenImage21
        vae = Mock(config=SimpleNamespace(scaling_factor=1.0))
        with (
            patch.object(model, "_single_file_checkpoint_path", return_value=None),
            patch.object(model, "_ramtorch_vae_requested", return_value=False),
            patch("simpletuner.helpers.models.common.deepspeed_zero_init_disabled_context_manager", return_value=[]),
            patch.object(AutoencoderKLQwenImage21, "from_pretrained", return_value=vae) as load,
        ):
            for _ in range(2):
                model.load_vae(move_to_device=False)
                self.assertIs(model.vae, vae)
                load.assert_called_with(**model._get_vae_load_kwargs())
            self.assertEqual(load.call_count, 2)
            self.assertEqual(vae.enable_tiling.call_count, 2)
            self.assertEqual(vae.enable_slicing.call_count, 2)
