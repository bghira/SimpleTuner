import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from simpletuner.helpers.assistant_lora import load_assistant_adapter
from simpletuner.helpers.distillation.assistant_lora.distiller import AssistantLoRADistiller
from simpletuner.helpers.models.common import PredictionTypes
from simpletuner.helpers.models.flux3_image import image_ids, patchify, text_ids, unpatchify
from simpletuner.helpers.models.flux3_image.model import Flux3Image
from simpletuner.helpers.models.flux3_image.pipeline import FLOW_SHIFT, Flux3ImagePipeline, guided_velocity
from simpletuner.helpers.models.flux3_image.text import collate_embeddings
from simpletuner.helpers.models.flux3_image.transformer import Flux3ImageTransformer2DModel
from simpletuner.helpers.training.custom_schedule import TwinFlowScheduler
from simpletuner.helpers.training.layersync import LayerSyncRegularizer
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer


def tiny_transformer(**kwargs):
    config = dict(
        in_channels=8,
        cond_in_channels=8,
        joint_attention_dim=16,
        num_attention_heads=2,
        attention_head_dim=8,
        axes_dims_rope=(2, 2, 2, 2),
        num_layers=1,
        num_single_layers=2,
        timestep_channels=8,
    )
    config.update(kwargs)
    return Flux3ImageTransformer2DModel(**config)


def text_only_ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.distributed.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(42)
        model = tiny_transformer()
        model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=Flux3Image.DEFAULT_LORA_TARGET))
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=Flux3Image.DDP_FIND_UNUSED_PARAMETERS
        )
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        for _ in range(2):
            loss = (
                wrapped(
                    torch.randn(1, 4, 8),
                    torch.randn(1, 3, 16),
                    torch.tensor([0.5]),
                    img_ids=image_ids(2, 2, "cpu"),
                    return_dict=False,
                )[0]
                .square()
                .mean()
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    finally:
        torch.distributed.destroy_process_group()


def mathematical_reference(
    model, image, text, timestep, positions, reference=None, reference_positions=None, destination=None
):
    weights = model.state_dict()
    width = model.inner_dim
    heads = model.config.num_attention_heads
    head_width = model.config.attention_head_dim

    def linear(value, name):
        return F.linear(value, weights[name + ".weight"])

    def time_embedding(time, prefix="timestep_embedder"):
        scaled = time.to(image.dtype) * 1000
        half = model.config.timestep_channels // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        phase = scaled.float()[..., None] * frequencies
        encoded = torch.cat((phase.cos(), phase.sin()), -1).to(image.dtype)
        return linear(F.silu(linear(encoded, prefix + ".linear_1")), prefix + ".linear_2")

    def rotate(value, ids):
        phases = []
        for axis, count in enumerate(model.config.axes_dims_rope):
            frequencies = 10000 ** (-torch.arange(0, count, 2).double() / count)
            phases.append(ids[:, axis, None].double() * frequencies)
        angles = torch.cat(phases, -1)
        rotations = torch.polar(torch.ones_like(angles), angles).to(torch.complex64)
        complex_values = torch.view_as_complex(value.float().unflatten(-1, (-1, 2)).contiguous())
        rotated = complex_values * rotations[None, :, None]
        return torch.view_as_real(rotated).flatten(-2).to(value.dtype)

    def block(value, embedding, ids, name):
        shift, scale, gate = embedding.chunk(3, -1)
        normalized = F.layer_norm(value, (width,), eps=model.config.eps) * (1 + scale) + shift
        query = linear(normalized, name + ".attn.to_q").unflatten(-1, (heads, head_width))
        key = linear(normalized, name + ".attn.to_k").unflatten(-1, (heads, head_width))
        val = linear(normalized, name + ".attn.to_v").unflatten(-1, (heads, head_width))
        query = F.rms_norm(query.float(), (head_width,), weights[name + ".attn.norm_q.weight"].float(), model.config.eps)
        key = F.rms_norm(key.float(), (head_width,), weights[name + ".attn.norm_k.weight"].float(), model.config.eps)
        query, key = rotate(query, ids).to(val.dtype), rotate(key, ids).to(val.dtype)
        attention = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), val.transpose(1, 2))
        attention = linear(attention.transpose(1, 2).flatten(-2), name + ".attn.to_out.0")
        a, b = linear(normalized, name + ".ff.linear_in").chunk(2, -1)
        feedforward = linear(F.silu(a) * b, name + ".ff.linear_out")
        return value + gate * (attention + feedforward)

    current = time_embedding(timestep[:, None])
    clean = time_embedding(torch.zeros_like(timestep[:, None]))
    if destination is not None:
        target = time_embedding(destination[:, None], "target_timestep_embedder")
        current, clean = current + target, clean + target
    streams = {"txt": linear(text, "context_embedder"), "image": linear(image, "x_embedders.image")}
    embeddings = {"txt": clean, "image": current}
    ids = {"txt": text_ids(text.shape[1], image.device), "image": positions}
    if reference is not None:
        streams["image_cond"] = linear(reference, "x_embedders.image_cond")
        embeddings["image_cond"], ids["image_cond"] = clean, reference_positions
    for mode in streams:
        modulation = linear(F.silu(embeddings[mode]), "early_stream_modulations." + mode + ".linear")
        prefix = "txt_mode_blocks" if mode == "txt" else "mode_blocks." + mode
        for index in range(model.config.num_layers):
            streams[mode] = block(streams[mode], modulation, ids[mode], prefix + "." + str(index))
    modulation = torch.cat(
        [
            linear(F.silu(embeddings[mode]), "single_stream_modulations." + mode + ".linear").expand(-1, value.shape[1], -1)
            for mode, value in streams.items()
        ],
        1,
    )
    combined = torch.cat(list(streams.values()), 1)
    all_ids = torch.cat(list(ids.values()), 0)
    for index in range(model.config.num_single_layers):
        combined = block(combined, modulation, all_ids, "single_transformer_blocks." + str(index))
    generated = combined[:, text.shape[1] : text.shape[1] + image.shape[1]]
    scale, shift = linear(F.silu(current), "norm_out.linear").chunk(2, -1)
    output = F.layer_norm(generated, (width,), eps=model.config.eps) * (1 + scale) + shift
    return linear(output, "proj_out")


class Flux3ImageTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = tiny_transformer()
        self.image = torch.randn(2, 4, 8)
        self.text = torch.randn(2, 3, 16)
        self.time = torch.tensor([0.23, 0.81])
        self.ids = image_ids(2, 2, "cpu")

    def test_matches_published_checkpoint_names_and_shapes(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/flux3_image_transformer.json").read_text())
        with torch.device("meta"):
            model = Flux3ImageTransformer2DModel.from_config(fixture["config"])
        actual = {key: list(value.shape) for key, value in model.state_dict().items()}
        self.assertEqual(actual, fixture["shapes"])
        self.assertEqual(sum(value.numel() for value in model.parameters()), 5538458368)

    def test_forward_matches_independent_mathematical_reference(self):
        for conditioned in (False, True):
            with self.subTest(conditioned=conditioned):
                ref = torch.randn(2, 6, 8) if conditioned else None
                ref_ids = image_ids(2, 3, "cpu", 0) if conditioned else None
                actual = self.model(
                    self.image,
                    self.text,
                    self.time,
                    img_ids=self.ids,
                    conditioning_hidden_states=ref,
                    conditioning_ids=ref_ids,
                ).sample
                expected = mathematical_reference(self.model, self.image, self.text, self.time, self.ids, ref, ref_ids)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_distilled_destination_embedding_matches_reference(self):
        model = tiny_transformer(flow_map=True)
        destination = torch.tensor([0.1, 0.5])
        actual = model(self.image, self.text, self.time, img_ids=self.ids, target_timestep=destination).sample
        expected = mathematical_reference(model, self.image, self.text, self.time, self.ids, destination=destination)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        scalar = model(self.image, self.text, self.time, img_ids=self.ids, target_timestep=0.0).sample
        self.assertEqual(scalar.shape, self.image.shape)
        single = model(
            self.image[:1],
            self.text[:1],
            self.time[:1],
            img_ids=self.ids,
            target_timestep=torch.tensor([[0.1]]),
        ).sample
        torch.testing.assert_close(single, expected[:1], atol=2e-6, rtol=2e-5)
        with self.assertRaisesRegex(ValueError, "destination"):
            model(self.image, self.text, self.time, img_ids=self.ids)

    def test_attention_keeps_samples_independent(self):
        first = self.model(self.image, self.text, self.time, img_ids=self.ids).sample[0]
        self.image[1].mul_(100)
        self.text[1].mul_(100)
        second = self.model(self.image, self.text, self.time, img_ids=self.ids).sample[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_batch_padding_does_not_change_a_shorter_prompt(self):
        individual = self.model(self.image[:1], self.text[:1], self.time[:1], img_ids=self.ids).sample
        padded = torch.cat((self.text[:1], torch.randn(1, 5, 16) * 100), 1)
        masked = self.model(
            self.image[:1],
            padded,
            self.time[:1],
            img_ids=self.ids,
            encoder_attention_mask=torch.tensor([[True] * 3 + [False] * 5]),
        ).sample
        torch.testing.assert_close(individual, masked, rtol=2e-5, atol=2e-6)

    def test_layer_capture_excludes_reference_and_text_tokens_and_retains_gradients(self):
        states = HiddenStateBuffer(capture_layers={0, 2})
        self.model(
            self.image,
            self.text,
            self.time,
            img_ids=self.ids,
            conditioning_hidden_states=torch.randn(2, 6, 8),
            conditioning_ids=image_ids(2, 3, "cpu", 0),
            hidden_states_buffer=states,
        )
        self.assertEqual(set(states), {"layer_0", "layer_2"})
        self.assertEqual(states["layer_2"].shape, (2, 4, 16))
        regularizer = LayerSyncRegularizer(
            SimpleNamespace(layersync_enabled=True, layersync_student_block=2, layersync_teacher_block=0)
        )
        loss, _ = regularizer.compute_loss(states)
        loss.backward()
        self.assertGreater(self.model.single_transformer_blocks[-1].ff.linear_in.weight.grad.abs().sum().item(), 0)

    def test_checkpointed_training_matches_eager_gradients(self):
        eager = self.model(self.image, self.text, self.time, img_ids=self.ids).sample
        eager.square().mean().backward()
        gradients = {name: p.grad.clone() for name, p in self.model.named_parameters() if p.grad is not None}
        self.model.zero_grad(set_to_none=True)
        self.model.enable_gradient_checkpointing()
        actual = self.model(self.image, self.text, self.time, img_ids=self.ids).sample
        actual.square().mean().backward()
        torch.testing.assert_close(actual, eager)
        for name, value in self.model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(value.grad, gradients[name])

    def test_segmented_checkpointing_matches_eager_outputs_and_all_gradients(self):
        for flow_map in (False, True):
            for conditioned in (False, True):
                model = tiny_transformer(num_layers=5, num_single_layers=7, flow_map=flow_map)
                inputs = {
                    "hidden_states": self.image.clone().requires_grad_(),
                    "encoder_hidden_states": self.text.clone().requires_grad_(),
                    "timestep": self.time,
                    "img_ids": self.ids,
                    "encoder_attention_mask": torch.tensor([[True, True, False], [True, True, True]]),
                }
                if conditioned:
                    inputs.update(
                        conditioning_hidden_states=torch.randn(2, 6, 8, requires_grad=True),
                        conditioning_ids=image_ids(2, 3, "cpu", 0),
                        conditioning_attention_mask=torch.tensor([[True] * 5 + [False], [True] * 6]),
                    )
                if flow_map:
                    inputs["target_timestep"] = torch.tensor([0.1, 0.5])
                expected = model(**inputs).sample
                expected.square().mean().backward()
                parameter_gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                input_gradients = {name: value.grad.clone() for name, value in inputs.items() if value.requires_grad}
                model.enable_gradient_checkpointing()
                for interval, stride in ((2, None), (2, 4), (3, 5)):
                    with self.subTest(flow_map=flow_map, conditioned=conditioned, interval=interval, stride=stride):
                        model.zero_grad(set_to_none=True)
                        for value in inputs.values():
                            if value.requires_grad:
                                value.grad = None
                        model.set_gradient_checkpointing_interval(interval)
                        model.set_gradient_checkpointing_segment_stride(stride)
                        actual = model(**inputs).sample
                        actual.square().mean().backward()
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(
                            {name for name, p in model.named_parameters() if p.grad is not None},
                            set(parameter_gradients),
                        )
                        for name, parameter in model.named_parameters():
                            if name in parameter_gradients:
                                torch.testing.assert_close(parameter.grad, parameter_gradients[name])
                        for name, gradient in input_gradients.items():
                            torch.testing.assert_close(inputs[name].grad, gradient)

    def test_segmented_checkpoint_calls_group_adjacent_blocks_in_every_stream(self):
        for stride in (None, 4):
            with self.subTest(stride=stride):
                model = tiny_transformer(num_layers=5, num_single_layers=5)
                model.enable_gradient_checkpointing()
                model.set_gradient_checkpointing_interval(2)
                model.set_gradient_checkpointing_segment_stride(stride)
                groups = []
                active_group = None

                def record_block(name):
                    def hook(_module, _args):
                        if active_group is not None:
                            active_group.append(name)

                    return hook

                stacks = {
                    "text": model.txt_mode_blocks,
                    "image": model.mode_blocks.image,
                    "reference": model.mode_blocks.image_cond,
                    "joint": model.single_transformer_blocks,
                }
                for name, blocks in stacks.items():
                    for index, block in enumerate(blocks):
                        block.register_forward_pre_hook(record_block((name, index)))

                def record_checkpoint(function, *args, **kwargs):
                    nonlocal active_group
                    active_group = []
                    groups.append(active_group)
                    result = torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
                    active_group = None
                    return result

                model._checkpoint_fn = record_checkpoint
                model(
                    self.image,
                    self.text,
                    self.time,
                    img_ids=self.ids,
                    conditioning_hidden_states=torch.randn(2, 6, 8),
                    conditioning_ids=image_ids(2, 3, "cpu", 0),
                )
                starts = (0, 2, 4) if stride is None else (0, 4)
                expected = [
                    [(name, index) for index in range(start, min(start + 2, 5))] for name in stacks for start in starts
                ]
                self.assertEqual(groups, expected)

    def test_frozen_lora_segments_preserve_gradients_without_grad_inputs(self):
        for dtype in (torch.float32, torch.bfloat16):
            model = tiny_transformer(num_layers=3, num_single_layers=5).to(dtype)
            model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=Flux3Image.DEFAULT_LORA_TARGET))
            for name, parameter in model.named_parameters():
                parameter.requires_grad_("lora_" in name)
                if "lora_B" in name:
                    with torch.no_grad():
                        parameter.normal_(std=0.02)
            inputs = {
                "hidden_states": self.image.to(dtype),
                "encoder_hidden_states": self.text.to(dtype),
                "timestep": self.time.to(dtype),
                "img_ids": self.ids,
                "conditioning_hidden_states": torch.randn(2, 6, 8, dtype=dtype),
                "conditioning_ids": image_ids(2, 3, "cpu", 0),
            }
            expected = model(**inputs).sample
            expected.float().square().mean().backward()
            gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            model.enable_gradient_checkpointing()
            for interval, stride in ((2, None), (2, 4)):
                with self.subTest(dtype=dtype, interval=interval, stride=stride):
                    model.zero_grad(set_to_none=True)
                    model.set_gradient_checkpointing_interval(interval)
                    model.set_gradient_checkpointing_segment_stride(stride)
                    actual = model(**inputs).sample
                    actual.float().square().mean().backward()
                    torch.testing.assert_close(actual, expected)
                    self.assertEqual({name for name, p in model.named_parameters() if p.grad is not None}, set(gradients))
                    for name, parameter in model.named_parameters():
                        if name in gradients:
                            torch.testing.assert_close(parameter.grad, gradients[name])

    def test_segmented_checkpointing_preserves_layer_capture_fallback(self):
        self.model.enable_gradient_checkpointing()
        self.model.set_gradient_checkpointing_interval(2)
        self.model.set_gradient_checkpointing_segment_stride(4)
        with patch(
            "simpletuner.helpers.models.flux3_image.transformer.checkpoint_sequential_state",
            side_effect=AssertionError("Layer capture must use the per-block path"),
        ):
            self.test_layer_capture_excludes_reference_and_text_tokens_and_retains_gradients()

    def test_segmented_checkpointing_is_not_used_without_gradients(self):
        self.model.enable_gradient_checkpointing()
        self.model.set_gradient_checkpointing_interval(2)
        with (
            torch.no_grad(),
            patch(
                "simpletuner.helpers.models.flux3_image.transformer.checkpoint_sequential_state",
                side_effect=AssertionError("Inference must not checkpoint"),
            ),
        ):
            result = self.model(self.image, self.text, self.time, img_ids=self.ids).sample
        self.assertEqual(result.shape, self.image.shape)

    def test_regional_compilation_with_segments_preserves_gradients(self):
        from accelerate.utils.other import compile_regions

        model = tiny_transformer(num_layers=3, num_single_layers=5)
        expected = model(self.image, self.text, self.time, img_ids=self.ids).sample
        expected.square().mean().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.enable_gradient_checkpointing()
        model.set_gradient_checkpointing_interval(2)
        model.set_gradient_checkpointing_segment_stride(4)
        model.zero_grad(set_to_none=True)
        compiled = compile_regions(model, backend="eager")
        actual = compiled(self.image, self.text, self.time, img_ids=self.ids).sample
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name])

    def test_accelerate_regional_compilation_preserves_container_access_and_gradients(self):
        from accelerate.utils.other import compile_regions

        model = tiny_transformer(num_layers=2)
        expected = model(self.image, self.text, self.time, img_ids=self.ids).sample
        expected.square().mean().backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        compiled = compile_regions(model, backend="eager")
        actual = compiled(self.image, self.text, self.time, img_ids=self.ids).sample
        actual.square().mean().backward()
        torch.testing.assert_close(actual, expected)
        for name, parameter in model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(parameter.grad, gradients[name])

    def test_latent_layout_and_position_ids(self):
        raw = torch.randn(2, 32, 4, 6)
        torch.testing.assert_close(unpatchify(patchify(raw)), raw, rtol=0, atol=0)
        ids = image_ids(2, 3, "cpu", reference_index=2)
        self.assertEqual(ids[0].tolist(), [0, 0, 0, -3])
        self.assertEqual(ids[-1].tolist(), [0, 0.5, 1, -3])

    def test_cached_text_collation_preserves_existing_batches_and_masks(self):
        first = {"prompt_embeds": torch.randn(2, 80, 16), "attention_masks": torch.ones(2, 80, dtype=torch.bool)}
        second = {"prompt_embeds": torch.randn(160, 16), "attention_masks": torch.ones(160, dtype=torch.bool)}
        output = collate_embeddings([first, second])
        self.assertEqual(output["prompt_embeds"].shape, (3, 160, 16))
        self.assertEqual(output["attention_masks"].sum(1).tolist(), [80, 80, 160])
        torch.testing.assert_close(output["prompt_embeds"][:2, :80], first["prompt_embeds"])

    def test_distilled_training_predicts_instantaneous_velocity(self):
        foundation = Flux3Image.__new__(Flux3Image)
        foundation.config = SimpleNamespace(weight_dtype=torch.float32)
        foundation.accelerator = SimpleNamespace(device=torch.device("cpu"))
        foundation.model = tiny_transformer(flow_map=True)
        foundation.PREDICTION_TYPE = PredictionTypes.FLOW_MAP
        foundation.unwrap_model = lambda model: model
        foundation._new_hidden_state_buffer = lambda: HiddenStateBuffer(capture_layers={1})
        foundation.crepa_regularizer = SimpleNamespace(block_index=1)
        batch = {
            "noisy_latents": torch.randn(2, 8, 2, 2),
            "latents": torch.randn(2, 8, 2, 2),
            "noise": torch.randn(2, 8, 2, 2),
            "timesteps": torch.tensor([250.0, 750.0]),
            "prompt_embeds": torch.randn(2, 3, 16),
        }
        captured = {}
        hook = foundation.model.register_forward_pre_hook(
            lambda module, args, kwargs: captured.update(kwargs), with_kwargs=True
        )
        prediction = foundation.model_predict(batch)
        hook.remove()
        torch.testing.assert_close(captured["timestep"], torch.tensor([[0.25], [0.75]]))
        torch.testing.assert_close(captured["target_timestep"], captured["timestep"])
        target = foundation.get_prediction_target(batch)
        torch.testing.assert_close(target, batch["noise"] - batch["latents"])
        self.assertEqual(prediction["crepa_hidden_states"].shape, (2, 4, 16))
        F.mse_loss(prediction["model_prediction"], target).backward()
        self.assertIsNotNone(foundation.model.target_timestep_embedder.linear_1.weight.grad)

    def test_family_defaults_apply_before_creating_the_schedule(self):
        from simpletuner.helpers.configuration.cmd_args import get_default_config

        for flavour, prediction_type in ((None, PredictionTypes.FLOW_MATCHING), ("klein", PredictionTypes.FLOW_MAP)):
            config = SimpleNamespace(
                **dict(
                    get_default_config(),
                    model_family="flux3_image",
                    model_flavour=flavour,
                    pretrained_model_name_or_path=None,
                    model_type="lora",
                    vae_path=None,
                )
            )
            foundation = Flux3Image(config, SimpleNamespace(device=torch.device("cpu")))
            foundation.check_user_config()
            self.assertEqual(config.aspect_bucket_alignment, 16)
            self.assertEqual(config.vae_dtype, "fp32")
            self.assertEqual(config.tokenizer_max_length, 8192)
            self.assertEqual(foundation.noise_schedule.config.shift, FLOW_SHIFT)
            self.assertIs(foundation.PREDICTION_TYPE, prediction_type)

    def test_ragged_reference_batches_match_individual_predictions(self):
        foundation = Flux3Image.__new__(Flux3Image)
        foundation.config = SimpleNamespace(weight_dtype=torch.float32)
        foundation.accelerator = SimpleNamespace(device=torch.device("cpu"))
        foundation.LATENT_CHANNEL_COUNT = 8
        references = [torch.randn(8, 2, 3), torch.randn(8, 3, 3)]
        batch = foundation.prepare_batch_conditions(
            {
                "latents": torch.randn(2, 8, 2, 2),
                "conditioning_latents": [references],
            },
            {},
        )
        self.assertEqual(batch["conditioning_attention_mask"].sum(1).tolist(), [6, 9])
        image, text, time = torch.randn(2, 4, 8), torch.randn(2, 3, 16), torch.tensor([0.2, 0.8])
        predicted = self.model(
            image,
            text,
            time,
            img_ids=image_ids(2, 2, "cpu"),
            conditioning_hidden_states=batch["conditioning_packed_latents"],
            conditioning_ids=batch["conditioning_ids"],
            conditioning_attention_mask=batch["conditioning_attention_mask"],
        ).sample
        for index, latent in enumerate(references):
            individual = self.model(
                image[index : index + 1],
                text[index : index + 1],
                time[index : index + 1],
                img_ids=image_ids(2, 2, "cpu"),
                conditioning_hidden_states=latent.flatten(1).T.unsqueeze(0),
                conditioning_ids=image_ids(*latent.shape[-2:], "cpu", reference_index=0),
            ).sample
            torch.testing.assert_close(predicted[index : index + 1], individual, atol=1e-6, rtol=1e-5)

    def test_assistant_adapter_loads_frozen_alongside_a_trainable_adapter(self):
        config = LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_v"])
        self.model.add_adapter(config, adapter_name="default")
        state = get_peft_model_state_dict(self.model, adapter_name="default")
        with tempfile.TemporaryDirectory() as directory:
            Flux3ImagePipeline.save_lora_weights(directory, transformer_lora_layers=state)
            recipient = tiny_transformer(flow_map=True)
            self.assertTrue(
                load_assistant_adapter(
                    transformer=recipient, pipeline_cls=Flux3ImagePipeline, lora_path=directory, adapter_name="assistant"
                )
            )
            assistant = [p for name, p in recipient.named_parameters() if ".assistant." in name]
            self.assertTrue(assistant)
            self.assertFalse(any(p.requires_grad for p in assistant))
            recipient.add_adapter(config, adapter_name="default")
            self.assertTrue(any(p.requires_grad for name, p in recipient.named_parameters() if ".default." in name))

    def test_twinflow_sign_module_initializes_neutrally_and_is_saved_with_lora(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            model = Flux3ImageTransformer2DModel.from_pretrained(directory, enable_time_sign_embed=True)
        torch.testing.assert_close(
            model.time_sign_embed.weight, torch.zeros_like(model.time_sign_embed.weight), atol=0, rtol=0
        )
        model.add_adapter(
            LoraConfig(
                r=2,
                lora_alpha=2,
                target_modules=Flux3Image.DEFAULT_LORA_TARGET,
                modules_to_save=["time_sign_embed"],
            )
        )
        result = model(self.image, self.text, self.time, img_ids=self.ids, timestep_sign=torch.tensor([-1, 1])).sample
        result.square().mean().backward()
        state = get_peft_model_state_dict(model)
        self.assertIn("time_sign_embed.weight", state)
        self.assertTrue(any(p.grad is not None for p in model.time_sign_embed.parameters()))


class Flux3ImagePipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(321)
        self.model = tiny_transformer()
        self.pipeline = Flux3ImagePipeline(self.model)
        self.pipeline.set_progress_bar_config(disable=True)
        self.embeddings = torch.randn(2, 3, 16)

    def sample(self, **kwargs):
        args = dict(
            prompt_embeds=self.embeddings,
            height=32,
            width=32,
            num_inference_steps=4,
            guidance_scale=1.0,
            output_type="latent",
            generator=[torch.Generator().manual_seed(42), torch.Generator().manual_seed(43)],
        )
        args.update(kwargs)
        return self.pipeline(**args).images

    def test_official_static_schedule_and_seed_reproducibility(self):
        first, second = self.sample(), self.sample()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        grid = torch.linspace(1, 0, 5)
        expected = FLOW_SHIFT * grid / (1 + (FLOW_SHIFT - 1) * grid)
        torch.testing.assert_close(self.pipeline.scheduler.sigmas.cpu(), expected)
        self.assertEqual(first.dtype, torch.float32)

    def test_cfg_and_zero_star_have_expected_velocity_convention(self):
        positive, negative = torch.tensor([[[2.0, 4.0]]]), torch.tensor([[[1.0, 2.0]]])
        torch.testing.assert_close(guided_velocity(positive, negative, 3.5), negative + 3.5 * (positive - negative))
        torch.testing.assert_close(guided_velocity(positive, negative, 3.5, True), positive)
        out = self.sample(negative_prompt_embeds=torch.randn(2, 5, 16), guidance_scale=3.5, use_cfg_zero_star=True)
        self.assertTrue(out.isfinite().all())

    def test_twinflow_uses_supplied_generators(self):
        self.pipeline.scheduler = TwinFlowScheduler(stochast_ratio=0.7)
        first, second = self.sample(), self.sample()
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_step_callbacks_receive_only_requested_tensors_and_can_replace_latents(self):
        for inputs in ((), ("latents",)):
            with self.subTest(inputs=inputs):
                calls = []

                def callback(pipeline, index, timestep, values):
                    self.assertIs(pipeline, self.pipeline)
                    self.assertEqual(set(values), set(inputs))
                    calls.append(index)
                    return {"latents": torch.zeros_like(values["latents"])} if inputs else {}

                result = self.sample(callback_on_step_end=callback, callback_on_step_end_tensor_inputs=inputs)
                self.assertEqual(calls, [0, 1, 2, 3])
                if inputs:
                    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)

    def test_distilled_sampling_passes_destination_time(self):
        self.pipeline = Flux3ImagePipeline(tiny_transformer(flow_map=True))
        self.pipeline.set_progress_bar_config(disable=True)
        result = self.sample()
        self.assertTrue(result.isfinite().all())

    def test_pipeline_roundtrip_preserves_optional_components_and_scheduler(self):
        self.pipeline.scheduler = TwinFlowScheduler(stochast_ratio=0.7, shift=2.5)
        with tempfile.TemporaryDirectory() as directory:
            self.pipeline.save_pretrained(directory)
            self.pipeline = Flux3ImagePipeline.from_pretrained(directory, torch_dtype=torch.float32)
        self.pipeline.set_progress_bar_config(disable=True)
        self.assertIsInstance(self.pipeline.scheduler, TwinFlowScheduler)
        self.assertEqual(self.pipeline.scheduler.config.shift, 2.5)
        self.assertIsNone(self.pipeline.vae)
        self.assertTrue(self.sample().isfinite().all())

    @unittest.skipUnless(torch.backends.mps.is_available() or torch.cuda.is_available(), "Accelerator unavailable")
    def test_accelerator_twinflow_accepts_cpu_generators_and_backpropagates(self):
        device = "cuda" if torch.cuda.is_available() else "mps"
        self.pipeline.to(device)
        self.pipeline.scheduler = TwinFlowScheduler(stochast_ratio=0.7)
        first, second = self.sample(), self.sample()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        result = self.model(
            torch.randn(2, 4, 8, device=device),
            self.embeddings.to(device),
            torch.tensor([0.25, 0.75], device=device),
            img_ids=image_ids(2, 2, device),
        ).sample
        result.square().mean().backward()
        self.assertTrue(self.model.proj_out.weight.grad.isfinite().all())


class Flux3ImageIntegrationTests(unittest.TestCase):
    def test_validation_precaches_empty_negative_before_unloading_text_encoder(self):
        from simpletuner.helpers.configuration.cmd_args import get_default_config
        from simpletuner.helpers.training.validation import prepare_validation_prompt_list

        for negative in (None, ""):
            with self.subTest(negative=negative):
                config = SimpleNamespace(
                    **dict(
                        get_default_config(),
                        model_family="flux3_image",
                        model_flavour="klein-base",
                        model_type="lora",
                        pretrained_model_name_or_path=None,
                        vae_path=None,
                        validation_prompt="a bicycle",
                        validation_prompt_library=False,
                        validation_disable_unconditional=True,
                        validation_negative_prompt=negative,
                    )
                )
                foundation = Flux3Image(config, SimpleNamespace(device=torch.device("cpu")))
                cache = SimpleNamespace(
                    model_type="flux3_image",
                    text_cache_ondemand=False,
                    _requires_path_based_keys=False,
                    compute_embeddings_for_prompts=Mock(),
                )
                with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_args", return_value=config):
                    prepare_validation_prompt_list(config, cache, foundation)
                cache.compute_embeddings_for_prompts.assert_any_call(
                    [""],
                    is_validation=True,
                    load_from_cache=False,
                    is_negative_prompt=True,
                )

    def test_full_resume_restores_training_projector_and_parameter_identity(self):
        from simpletuner.helpers.training.save_hooks import SaveHookManager

        for shard_size in ("10GB", "10KB"):
            with self.subTest(shard_size=shard_size), tempfile.TemporaryDirectory() as directory:
                model = tiny_transformer()
                model.crepa_projector = torch.nn.Sequential(torch.nn.LayerNorm(16), torch.nn.Linear(16, 8))
                expected = {key: value.clone() for key, value in model.state_dict().items()}
                identities = {key: id(value) for key, value in model.named_parameters()}
                model.save_pretrained(Path(directory) / "transformer", max_shard_size=shard_size)
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.zero_()
                manager = SaveHookManager.__new__(SaveHookManager)
                manager.args = SimpleNamespace(model_type="full", model_family="flux3_image")
                manager.denoiser_class = Flux3ImageTransformer2DModel
                manager.denoiser_subdir = "transformer"
                manager._load_full_model([model], directory)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                self.assertEqual({key: id(value) for key, value in model.named_parameters()}, identities)

    def test_inference_export_omits_training_projector_and_rejects_unknown_parameters(self):
        from simpletuner.helpers.training.save_hooks import (
            _build_model_from_state_dict_for_save,
            _get_fsdp2_pipeline_export_spec,
        )

        foundation = Flux3Image.__new__(Flux3Image)
        foundation.config = SimpleNamespace(controlnet=False)
        foundation.model = tiny_transformer()
        foundation.unwrap_model = lambda model=None: model or foundation.model
        foundation.model.crepa_projector = torch.nn.Linear(16, 8)
        spec = _get_fsdp2_pipeline_export_spec(foundation)
        exported = _build_model_from_state_dict_for_save(spec.model_class, spec.model_config, foundation.model.state_dict())
        self.assertFalse(hasattr(exported, "crepa_projector"))
        for name, parameter in exported.named_parameters():
            torch.testing.assert_close(parameter, foundation.model.state_dict()[name], atol=0, rtol=0)
        foundation.model.unexpected_head = torch.nn.Linear(16, 8)
        with self.assertRaisesRegex(RuntimeError, "cannot be reconstructed exactly"):
            _get_fsdp2_pipeline_export_spec(foundation)

    @unittest.skipUnless(torch.distributed.is_available() and torch.distributed.is_gloo_available(), "Gloo unavailable")
    def test_text_only_lora_completes_two_ddp_iterations(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(text_only_ddp_worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2)

    def test_assistant_rollout_prepares_a_real_flow_map_training_batch(self):
        from simpletuner.helpers.configuration.cmd_args import get_default_config

        config = SimpleNamespace(
            **dict(
                get_default_config(),
                model_family="flux3_image",
                model_flavour="klein",
                model_type="lora",
                pretrained_model_name_or_path=None,
                vae_path=None,
                weight_dtype=torch.float32,
                flow_matching=True,
            )
        )
        accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            process_index=0,
            num_processes=1,
            is_main_process=True,
            unwrap_model=lambda model, **kwargs: model,
        )
        foundation = Flux3Image(config, accelerator)
        foundation.check_user_config()
        foundation.model = tiny_transformer(in_channels=128, cond_in_channels=128, flow_map=True)
        foundation.model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=Flux3Image.DEFAULT_LORA_TARGET))
        distiller = AssistantLoRADistiller(
            foundation,
            config={
                "model_type": "lora",
                "num_inference_steps": 2,
                "resolutions": [[48, 32]],
            },
        )
        cached = {"prompt_embeds": torch.randn(2, 80, 16), "attention_masks": torch.ones(2, 80, dtype=torch.bool)}
        backend = {"text_embed_cache": SimpleNamespace(compute_embeddings_for_prompts=Mock(return_value=cached))}
        with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_data_backend", return_value=backend):
            batch = distiller.prepare_caption_batch(
                {"captions": ["a bicycle", "a fox"], "data_backend_id": "captions"}, foundation, {"global_step": 0}
            )
        self.assertEqual(batch["noisy_latents"].shape, (2, 128, 2, 3))
        prediction = foundation.model_predict(batch)["model_prediction"]
        F.mse_loss(prediction, foundation.get_prediction_target(batch)).backward()
        self.assertTrue(any(p.grad is not None for name, p in foundation.model.named_parameters() if "lora_B" in name))
        self.assertFalse(any(p.grad is not None for name, p in foundation.model.named_parameters() if "lora_" not in name))


if __name__ == "__main__":
    unittest.main()
