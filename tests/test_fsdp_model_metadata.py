import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.configuration.cmd_args import get_default_config
from simpletuner.helpers.models.common import ModelFoundation
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.trainer import Trainer


class FSDPModelMetadataTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_trainer(self, family="qwen_image", **overrides):
        config = get_default_config()
        config.update(
            model_family=family,
            model_flavour="v2.1" if family == "qwen_image" else None,
            pretrained_model_name_or_path="local-model",
            vae_path=None,
            fsdp_enable=True,
            fsdp_version=2,
            fsdp_auto_wrap_policy="transformer_based_wrap",
            fsdp_state_dict_type="SHARDED_STATE_DICT",
            fsdp_cpu_ram_efficient_loading=True,
            fsdp_cpu_offload=False,
            deepspeed_config=None,
        )
        config.update(overrides)
        trainer = object.__new__(Trainer)
        trainer.model = None
        trainer.config = SimpleNamespace(**config)
        plugin = trainer._prepare_fsdp_plugin_for_accelerator(trainer._load_fsdp_plugin())
        # Only scheduler I/O is irrelevant here; model selection uses the real constructor.
        with patch.object(ModelFoundation, "setup_training_noise_schedule"):
            trainer.model = ModelRegistry.get(family)(trainer.config, None)
        return trainer, plugin

    def test_qwen_selection_replaces_early_family_hints(self):
        for flavour, path, expected in (
            ("v2.0", None, "QwenImageTransformerBlock"),
            ("v2.1", None, "QwenImage21TransformerBlock"),
            (None, None, "QwenImage21TransformerBlock"),
            ("v2.1", "custom-model", "QwenImage21TransformerBlock"),
            ("v2.0", "custom-model", "QwenImageTransformerBlock"),
            (None, "custom-model", "QwenImageTransformerBlock"),
        ):
            for explicit in (None, expected):
                with self.subTest(flavour=flavour, path=path, explicit=explicit):
                    trainer, plugin = self.make_trainer(
                        model_flavour=flavour,
                        pretrained_model_name_or_path=path,
                        fsdp_transformer_layer_cls_to_wrap=explicit,
                    )
                    trainer._configure_fsdp_model_metadata(plugin)
                    self.assertEqual(plugin.transformer_cls_names_to_wrap, [expected])
                    self.assertEqual(trainer.config.fsdp_transformer_layer_cls_to_wrap, expected)
                    self.assertTrue(plugin.cpu_ram_efficient_loading)
                    trainer._configure_fsdp_model_metadata(plugin)
                    self.assertEqual(plugin.transformer_cls_names_to_wrap, [expected])

    def test_trainer_finalizes_metadata_after_selecting_model(self):
        for explicit in (None, "QwenImage21TransformerBlock"):
            with self.subTest(explicit=explicit):
                prepared, plugin = self.make_trainer(fsdp_transformer_layer_cls_to_wrap=explicit)
                self.assertIn("QwenImageTransformerBlock", plugin.transformer_cls_names_to_wrap)

                def parse_arguments(trainer, **kwargs):
                    trainer.config = prepared.config
                    trainer.accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
                    trainer._fsdp_configured_transformer_cls = prepared._fsdp_configured_transformer_cls

                with (
                    patch.object(Trainer, "parse_arguments", parse_arguments),
                    patch.object(Trainer, "_misc_init"),
                    patch.object(Trainer, "init_noise_schedule"),
                    patch.object(ModelFoundation, "setup_training_noise_schedule"),
                    patch.object(QwenImage, "check_user_config"),
                    patch("simpletuner.helpers.training.trainer.configure_inductor_wrapper"),
                    patch("simpletuner.helpers.training.trainer.DynamoCacheManager"),
                ):
                    trainer = Trainer()
                self.assertIs(trainer.accelerator.state.fsdp_plugin, plugin)
                self.assertEqual(plugin.transformer_cls_names_to_wrap, ["QwenImage21TransformerBlock"])

    def test_fixed_class_families_keep_their_hints(self):
        for family in ("flux", "sdxl", "sd3"):
            with self.subTest(family=family):
                trainer, plugin = self.make_trainer(family)
                before = list(plugin.transformer_cls_names_to_wrap)
                trainer._configure_fsdp_model_metadata(plugin)
                self.assertEqual(plugin.transformer_cls_names_to_wrap, before)
                if family == "sd3":
                    self.assertNotIn("PatchEmbed", before)
                    self.assertFalse(plugin.cpu_ram_efficient_loading)
                    self.assertFalse(trainer.config.fsdp_cpu_ram_efficient_loading)

    def test_training_target_uses_selected_class(self):
        for target, expected in (
            ("transformer", "MiniMaxMusic3TransformerBlock"),
            ("language_model", "Qwen3DecoderLayer"),
        ):
            with self.subTest(target=target):
                trainer, plugin = self.make_trainer("minimaxmusic", minimax_music_train_component=target)
                trainer._configure_fsdp_model_metadata(plugin)
                self.assertIn(expected, plugin.transformer_cls_names_to_wrap)
                self.assertEqual(
                    plugin.transformer_cls_names_to_wrap,
                    list(trainer.model.MODEL_CLASS._no_split_modules),
                )

    def test_execution_policies_survive_metadata_refresh(self):
        for version in (1, 2):
            for policy in ("transformer_based_wrap", "size_based_wrap", "no_wrap"):
                for ram in (False, True):
                    with self.subTest(version=version, policy=policy, ram=ram):
                        if version == 1 and ram and not torch.cuda.is_available():
                            self.skipTest("FSDP1 synchronized initialization requires an accelerator.")
                        trainer, plugin = self.make_trainer(
                            fsdp_version=version,
                            fsdp_reshard_after_forward="FULL_SHARD" if version == 1 else True,
                            fsdp_auto_wrap_policy=policy,
                            fsdp_cpu_ram_efficient_loading=ram,
                            fsdp_cpu_offload=True,
                            fsdp_activation_checkpointing=True,
                            gradient_checkpointing=True,
                        )
                        plugin.set_mixed_precision(torch.bfloat16)
                        attrs = ("auto_wrap_policy", "mixed_precision_policy", "cpu_offload", "state_dict_type")
                        before = {attr: getattr(plugin, attr) for attr in attrs}
                        trainer._configure_fsdp_model_metadata(plugin)
                        for attr, value in before.items():
                            self.assertIs(getattr(plugin, attr), value)
                        self.assertEqual(plugin.cpu_ram_efficient_loading, ram)
                        self.assertTrue(plugin.activation_checkpointing)
                        self.assertFalse(trainer.config.gradient_checkpointing)
                        self.assertEqual(plugin.transformer_cls_names_to_wrap, ["QwenImage21TransformerBlock"])

    def test_non_fsdp_does_not_create_plugin(self):
        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(fsdp_enable=False)
        self.assertIsNone(trainer._load_fsdp_plugin())


if __name__ == "__main__":
    unittest.main()
