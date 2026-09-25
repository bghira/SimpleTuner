import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from accelerate.utils import compile_regions
from diffusers import FlowMatchEulerDiscreteScheduler
from peft import LoraConfig, get_peft_model
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import AuxiliaryTrainingWrapper

from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.distillation.assistant_lora import AssistantLoRADistiller
from simpletuner.helpers.distillation.factory import DistillationMethod, DistillerFactory
from simpletuner.helpers.models.common import ImageModelFoundation, PipelineTypes
from simpletuner.helpers.models.generation import base_model_generation_context
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 4)

    def forward(self, value):
        return self.head(self.linear(value))


class BaseModelGenerationTests(unittest.TestCase):
    def test_teacher_disables_saved_modules_and_restores_mixed_state_after_failure(self):
        component = get_peft_model(TinyNetwork(), LoraConfig(r=2, target_modules=["linear"], modules_to_save=["head"]))
        model = SimpleNamespace(get_trained_component=lambda: component, accelerator=SimpleNamespace())
        inputs = torch.randn(2, 4)
        with component.disable_adapter():
            expected = component(inputs).detach()
        with torch.no_grad():
            component.base_model.model.head.modules_to_save["default"].weight.add_(3)
        component.train()
        component.base_model.model.linear.base_layer.eval()
        modules_before = [(module, module.training) for module in component.modules()]
        grad_before = [(parameter, parameter.requires_grad) for parameter in component.parameters()]
        with self.assertRaisesRegex(RuntimeError, "teacher failed"):
            with base_model_generation_context(model):
                self.assertFalse(torch.is_grad_enabled())
                torch.testing.assert_close(component(inputs), expected)
                for module in component.modules():
                    if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper)):
                        self.assertTrue(module.disable_adapters)
                raise RuntimeError("teacher failed")
        for module, training in modules_before:
            self.assertEqual(module.training, training)
        for parameter, requires_grad in grad_before:
            self.assertEqual(parameter.requires_grad, requires_grad)
            self.assertIsNone(parameter.grad)
        self.assertFalse(torch.equal(component(inputs), expected))

    def test_previously_disabled_adapter_and_lycoris_multiplier_are_restored(self):
        component = get_peft_model(TinyNetwork(), LoraConfig(r=2, target_modules=["linear"]))
        component.disable_adapter_layers()
        lycoris = SimpleNamespace(multiplier=0.4)
        lycoris.set_multiplier = lambda value: setattr(lycoris, "multiplier", value)
        model = SimpleNamespace(
            get_trained_component=lambda: component, accelerator=SimpleNamespace(_lycoris_wrapped_network=lycoris)
        )
        with base_model_generation_context(model):
            self.assertEqual(lycoris.multiplier, 0.0)
        self.assertEqual(lycoris.multiplier, 0.4)
        self.assertTrue(component.base_model.model.linear.disable_adapters)


class AssistantLoRADistillerTests(unittest.TestCase):
    def make_model(self, device="cpu"):
        torch.manual_seed(42)
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(
            model_flavour="v2.1",
            weight_dtype=torch.float32,
            flow_matching=True,
            input_perturbation=0,
            scheduled_sampling_max_step_offset=0,
        )
        model.accelerator = SimpleNamespace(device=torch.device(device), process_index=0, num_processes=1)
        model.vae_scale_factor = 16
        model.PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: QwenImage21Pipeline}
        model.diffusion_blocks_controller = None
        model.model = QwenImage21Transformer2DModel(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            context_in_dim=8,
            mlp_ratio=2,
            axes_dims_rope=(4, 6, 6),
        ).to(device)
        model.model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
        model.get_trained_component = lambda: model.model
        model.noise_schedule = FlowMatchEulerDiscreteScheduler(
            use_dynamic_shifting=True,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            base_shift=0.5,
            max_shift=0.9,
            shift_terminal=0.02,
        )
        model.sample_flow_sigmas = lambda batch, state: (
            torch.full((len(batch["latents"]), 1, 1, 1), 0.5, device=device),
            torch.full((len(batch["latents"]),), 500, device=device),
        )
        model._twinflow_active = lambda: False
        model._maybe_enable_reflexflow_default = lambda: None
        model._new_hidden_state_buffer = lambda: None
        model.prepare_batch_conditions = lambda batch, state: batch
        return model

    def make_distiller(self, model, **options):
        return AssistantLoRADistiller(
            model, config={"model_type": "lora", "num_inference_steps": 2, "resolutions": [[32, 32]], **options}
        )

    def caption_context(self, model):
        embeddings = {
            "prompt_embeds": torch.randn(1, 5, 8),
            "attention_masks": torch.ones(1, 5),
        }
        return patch(
            "simpletuner.helpers.distillation.assistant_lora.distiller.StateTracker.get_data_backend",
            return_value={"text_embed_cache": SimpleNamespace(compute_embeddings_for_prompts=Mock(return_value=embeddings))},
        )

    def test_native_qwen_rollout_and_training_backpropagate_only_adapter(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                model = self.make_model(device)
                distiller = self.make_distiller(model)
                self.assertIsNone(distiller.pipeline.vae)
                self.assertIsNone(distiller.pipeline.text_encoder)
                self.assertIsNone(distiller.pipeline.processor)
                scheduler = distiller.pipeline.scheduler
                with self.caption_context(model):
                    prepared = distiller.prepare_caption_batch(
                        {"captions": ["a fox"], "data_backend_id": "captions"}, model, {"global_step": 0}
                    )
                self.assertIs(distiller.pipeline.scheduler, scheduler)
                self.assertEqual(prepared["latents"].shape, (1, 4, 2, 2))
                self.assertFalse(prepared["latents"].requires_grad)
                self.assertFalse(prepared["latents"].is_inference())
                target = model.get_prediction_target(prepared)
                torch.testing.assert_close(target, prepared["noise"] - prepared["latents"])
                output = model._model_predict_21(prepared)["model_prediction"]
                loss = (output - target).square().mean()
                loss.backward()
                gradients = {
                    name: parameter.grad for name, parameter in model.model.named_parameters() if parameter.grad is not None
                }
                self.assertTrue(gradients)
                self.assertTrue(all("lora_" in name for name in gradients))
                self.assertTrue(any(gradient.abs().sum() > 0 for gradient in gradients.values()))
                self.assertFalse(distiller.requires_distillation_cache())
                self.assertIsNone(distiller.get_ode_generator_provider())
                returned, logs = distiller.compute_distill_loss(prepared, {}, loss)
                self.assertIs(returned, loss)
                self.assertEqual(logs, {})

    def test_fresh_targets_ignore_adapter_and_seed_counter_resumes(self):
        model = self.make_model()
        distiller = self.make_distiller(model, resolutions=[[32, 32], [64, 32]])
        batch = {"captions": ["a fox"], "data_backend_id": "captions"}
        with self.caption_context(model):
            first = distiller.prepare_caption_batch(batch, model, {"global_step": 0})
            with tempfile.TemporaryDirectory() as directory:
                distiller.on_save_checkpoint(1, directory)
                second = distiller.prepare_caption_batch(batch, model, {"global_step": 1})
                reloaded = self.make_distiller(model, resolutions=[[32, 32], [64, 32]])
                reloaded.on_load_checkpoint(directory)
                second_again = reloaded.prepare_caption_batch(batch, model, {"global_step": 1})
            torch.testing.assert_close(second["latents"], second_again["latents"])
            self.assertNotEqual(first["assistant_generation_seeds"], second["assistant_generation_seeds"])
            self.assertEqual(second["latents"].shape, (1, 4, 2, 4))
            with torch.no_grad():
                for name, parameter in model.model.named_parameters():
                    if "lora_B" in name:
                        parameter.add_(20)
            repeated = self.make_distiller(model).prepare_caption_batch(batch, model, {"global_step": 0})
            torch.testing.assert_close(first["latents"], repeated["latents"])

    def test_variable_batch_sizes_do_not_reuse_seeds(self):
        model = self.make_model()
        distiller = self.make_distiller(model)
        seeds = []
        for size in (3, 1, 2):
            embeddings = {"prompt_embeds": torch.randn(size, 5, 8)}
            with patch(
                "simpletuner.helpers.distillation.assistant_lora.distiller.StateTracker.get_data_backend",
                return_value={
                    "text_embed_cache": SimpleNamespace(compute_embeddings_for_prompts=Mock(return_value=embeddings))
                },
            ):
                prepared = distiller.prepare_caption_batch(
                    {"captions": ["a fox"] * size, "data_backend_id": "captions"}, model, {}
                )
            seeds.extend(prepared["assistant_generation_seeds"])
        self.assertEqual(seeds, list(range(42, 48)))

    def test_pipeline_failure_restores_scheduler_adapter_and_counter(self):
        model = self.make_model()
        distiller = self.make_distiller(model)
        scheduler = distiller.pipeline.scheduler
        model.model.train()
        with (
            self.caption_context(model),
            patch.object(QwenImage21Pipeline, "__call__", side_effect=RuntimeError("rollout failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "rollout failed"):
                distiller.prepare_caption_batch({"captions": ["a fox"], "data_backend_id": "captions"}, model, {})
        self.assertTrue(model.model.training)
        self.assertIs(distiller.pipeline.scheduler, scheduler)
        self.assertEqual(distiller._generation_index, 0)
        self.assertTrue(
            all(not module.disable_adapters for module in model.model.modules() if isinstance(module, BaseTunerLayer))
        )

    def test_teacher_uses_current_regionally_compiled_component_and_restores_state(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                model = self.make_model()
                distiller = self.make_distiller(model)
                original = model.model
                prepared = compile_regions(original, backend="eager")
                self.assertIsNot(prepared, original)
                self.assertIs(distiller.pipeline.transformer, original)
                model.model = prepared
                prepared.train()
                prepared.transformer_blocks[0].eval()
                training = [(module, module.training) for module in prepared.modules()]
                gradients = [(parameter, parameter.requires_grad) for parameter in prepared.parameters()]
                adapters = [module for module in prepared.modules() if isinstance(module, BaseTunerLayer)]
                scheduler = distiller.pipeline.scheduler
                calls = []

                def observe_teacher(module, args):
                    calls.append(module)
                    self.assertIs(module, prepared)
                    self.assertFalse(torch.is_grad_enabled())
                    self.assertTrue(all(adapter.disable_adapters for adapter in adapters))
                    self.assertTrue(all(not child.training for child in module.modules()))
                    if fail:
                        raise RuntimeError("prepared teacher failed")

                handle = prepared.register_forward_pre_hook(observe_teacher)
                try:
                    with self.caption_context(model):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "prepared teacher failed"):
                                distiller.prepare_caption_batch(
                                    {"captions": ["a fox"], "data_backend_id": "captions"}, model, {}
                                )
                        else:
                            batch = distiller.prepare_caption_batch(
                                {"captions": ["a fox"], "data_backend_id": "captions"}, model, {}
                            )
                            self.assertFalse(batch["latents"].requires_grad)
                finally:
                    handle.remove()
                self.assertTrue(calls)
                self.assertIs(distiller.pipeline.transformer, prepared)
                self.assertIs(distiller.pipeline.scheduler, scheduler)
                self.assertTrue(all(module.training == was_training for module, was_training in training))
                self.assertTrue(all(parameter.requires_grad == required for parameter, required in gradients))
                self.assertTrue(all(not adapter.disable_adapters for adapter in adapters))

    def test_resume_rejects_changed_generation_recipe(self):
        model = self.make_model()
        original = self.make_distiller(model)
        with tempfile.TemporaryDirectory() as directory:
            original.on_save_checkpoint(0, directory)
            for changes in ({"seed": 43}, {"resolutions": [[64, 32]]}, {"num_inference_steps": 3}):
                with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "requires unchanged"):
                    self.make_distiller(model, **changes).on_load_checkpoint(directory)

    def test_trainer_checkpoint_paths_restore_caption_and_generation_cursors_together(self):
        from simpletuner.helpers.data_backend.caption_dataset import CaptionDataset
        from simpletuner.helpers.data_backend.caption_sampler import CaptionSampler
        from simpletuner.helpers.training.caption_collate import collate_caption_batch
        from tests.helpers.data_backend.test_caption_pipeline import DummyCaptionMetadataBackend

        model = self.make_model()
        metadata = DummyCaptionMetadataBackend(3)
        dataset = CaptionDataset("caption", metadata)

        def make_sampler():
            return CaptionSampler("caption", metadata, model.accelerator, 1, shuffle=False)

        def generate(distiller, sampler):
            raw = collate_caption_batch([dataset[next(iter(sampler))]])
            return distiller.prepare_caption_batch(raw, model, {"global_step": 0})

        sampler = make_sampler()
        distiller = self.make_distiller(model)
        with tempfile.TemporaryDirectory() as directory, self.caption_context(model):
            generate(distiller, sampler)
            trainer_state = Path(directory, "training_state.json")
            trainer_state.write_text('{"global_step": 1}')
            distiller.on_save_checkpoint(1, directory)
            sampler.save_state(state_path=str(trainer_state))
            self.assertEqual(json.loads(trainer_state.read_text()), {"global_step": 1})
            saved_sampler = json.loads(Path(directory, "training_state-caption.json").read_text())
            self.assertEqual(saved_sampler["cursor"], 1)
            expected = generate(distiller, sampler)

            resumed_distiller = self.make_distiller(model)
            resumed_sampler = make_sampler()
            resumed_distiller.on_load_checkpoint(directory)
            resumed_sampler.load_states(state_path=str(trainer_state))
            resumed_sampler.log_state()
            actual = generate(resumed_distiller, resumed_sampler)
            self.assertEqual(actual["captions"], ["caption 1"])
            self.assertEqual(actual["assistant_generation_seeds"], expected["assistant_generation_seeds"])
            torch.testing.assert_close(actual["latents"], expected["latents"])

    def test_configuration_and_registration(self):
        self.assertEqual(DistillationMethod.from_string("assistant_lora"), DistillationMethod.ASSISTANT_LORA)
        model = self.make_model()
        for options in (
            {"model_type": "full"},
            {"num_inference_steps": 0},
            {"resolutions": []},
            {"resolutions": [[33, 32]]},
            {"resolutions": ["1024"]},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.make_distiller(model, **options)
        model.config.model_flavour = "v2.0"
        with self.assertRaisesRegex(NotImplementedError, "on-demand latent generation"):
            self.make_distiller(model)
        args = parse_cmdline_args(
            input_args=[
                "--model_family=qwen_image",
                "--model_flavour=v2.1",
                "--model_type=lora",
                "--optimizer=adamw_bf16",
                "--output_dir=output/assistant-test",
                "--data_backend_config=config/assistant-test.json",
                "--distillation_method=assistant_lora",
            ],
            exit_on_error=True,
        )
        self.assertEqual(args.distillation_method, "assistant_lora")

    def test_qwen21_loads_explicit_assistant_with_matching_pipeline(self):
        model = self.make_model()
        model.config.model_type = "lora"
        model.config.assistant_lora_path = "example/qwen21-assistant"
        model.assistant_adapter_name = "assistant"
        model.unwrap_model = lambda model: model
        with patch("simpletuner.helpers.assistant_lora.load_assistant_adapter", return_value=True) as loader:
            model._maybe_load_assistant_lora()
        self.assertIs(loader.call_args.kwargs["pipeline_cls"], QwenImage21Pipeline)
        self.assertTrue(model.assistant_lora_loaded)
        model.config.assistant_lora_path = None
        with patch("simpletuner.helpers.assistant_lora.load_assistant_adapter") as loader:
            model._maybe_load_assistant_lora()
        self.assertEqual(loader.call_args.kwargs["lora_path"], "SimpleTuner/Qwen-Image-2.1-training-assistant-v2")
        self.assertIs(loader.call_args.kwargs["pipeline_cls"], QwenImage21Pipeline)
        model.config.disable_assistant_lora = True
        with patch("simpletuner.helpers.assistant_lora.load_assistant_adapter") as loader:
            model._maybe_load_assistant_lora()
        loader.assert_not_called()
        self.assertFalse(QwenImage.supports_assistant_lora(SimpleNamespace(model_flavour="v1.0")))


class AssistantCaptionCacheTests(unittest.TestCase):
    def test_webui_caption_plan_accepts_assistant_distiller_and_requires_captions(self):
        from simpletuner.simpletuner_sdk.server.services.dataset_plan import compute_validations

        captions = {"id": "captions", "dataset_type": "caption", "type": "local", "instance_data_dir": "data/captions"}
        text_cache = {
            "id": "text",
            "dataset_type": "text_embeds",
            "type": "local",
            "default": True,
            "cache_dir": "cache/assistant",
        }
        kwargs = {"model_family": "qwen_image", "model_flavour": "v2.1", "distillation_method": "assistant_lora"}
        accepted = compute_validations([captions, text_cache], blueprints=[], **kwargs)
        self.assertFalse([message for message in accepted if message.level == "error"])
        rejected = compute_validations([text_cache], blueprints=[], **kwargs)
        self.assertTrue(any(message.level == "error" and "caption" in message.message for message in rejected))

    def test_caption_embeddings_are_precomputed_before_loader_registration(self):
        from simpletuner.helpers.data_backend.factory import FactoryRegistry
        from simpletuner.helpers.metadata.captions import CaptionRecord
        from tests.test_factory_caption_backend import _CaptionConfig, _CaptionMetadata

        events = []
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args = SimpleNamespace(distillation_method="assistant_lora", skip_file_discovery="caption")
        factory.accelerator = SimpleNamespace(is_local_main_process=True, num_processes=1)
        factory.caption_backends = {}
        metadata = _CaptionMetadata(events)
        metadata.caption_records = {"one": CaptionRecord("one", "a fox", "captions")}
        metadata._ordered_ids = ["one"]
        cache = SimpleNamespace(
            text_cache_ondemand=False,
            compute_embeddings_for_prompts=Mock(side_effect=lambda *args, **kwargs: events.append("cache")),
        )
        factory._assign_text_embed_cache = lambda backend, runtime: runtime.update(text_embed_cache=cache)
        factory._create_caption_dataloader = lambda *args: events.append("loader")
        backend = {"id": "captions", "type": "local", "dataset_type": "caption"}
        with (
            patch("simpletuner.helpers.data_backend.factory.create_backend_config", return_value=_CaptionConfig()),
            patch("simpletuner.helpers.data_backend.factory.init_backend_config", return_value={**backend, "config": {}}),
            patch(
                "simpletuner.helpers.data_backend.factory.build_backend_from_config",
                return_value={"data_backend": Mock(), "metadata_backend": metadata},
            ),
            patch("simpletuner.helpers.data_backend.factory.StateTracker") as tracker,
        ):
            factory._configure_caption_backend(backend)
        cache.compute_embeddings_for_prompts.assert_called_once_with(["a fox"], return_concat=False, load_from_cache=False)
        self.assertEqual(events, ["load", "cache", "loader"])
        self.assertIs(tracker.register_data_backend.call_args.args[0]["text_embed_cache"], cache)


if __name__ == "__main__":
    unittest.main()
