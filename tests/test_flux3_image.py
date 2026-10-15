import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from simpletuner.helpers.models.flux3_image import image_ids, patchify, text_ids, unpatchify
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
