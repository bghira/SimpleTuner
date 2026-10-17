import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from torch.nn import functional as F

from simpletuner.helpers.models.common import ImageModelFoundation, TextEmbedCacheKey
from simpletuner.helpers.models.moss_image.model import MOSSImage
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline
from simpletuner.helpers.models.moss_image.transformer import MOSSImageAttentionProcessor, _rotate
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer
from tests.test_moss_image import tiny_transformer


class MOSSImageEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)
        self.model = tiny_transformer()
        self.image = torch.randn(2, 8, 2, 3)
        self.text = torch.randn(2, 5, 16)
        self.mask = torch.tensor([[1, 1, 1, 1, 1], [1, 0, 1, 0, 0]], dtype=torch.bool)
        self.times = torch.tensor([700.0, 233.0])
        self.references = [[torch.randn(8, 2, 2), torch.randn(8, 1, 3)], [torch.randn(8, 3, 1)]]

    def predict(self, model=None, **kwargs):
        return (model or self.model)(
            self.image, self.times, self.text, self.mask, reference_latents=self.references, **kwargs
        ).sample

    def foundation(self, flavour="sft-edit"):
        foundation = object.__new__(MOSSImage)
        foundation.config = SimpleNamespace(
            model_flavour=flavour, weight_dtype=torch.float32, moss_image_prompt_upsample=False, moss_image_auto_json=False
        )
        foundation.accelerator = SimpleNamespace(device=torch.device("cpu"))
        foundation.LATENT_CHANNEL_COUNT = 8
        return foundation

    def test_empty_references_preserve_text_to_image(self):
        baseline = self.model(self.image, self.times, self.text, self.mask).sample
        actual = self.model(self.image, self.times, self.text, self.mask, reference_latents=[[], []]).sample
        torch.testing.assert_close(actual, baseline)

    def test_ragged_reference_batches_match_individual_samples_and_receive_gradients(self):
        self.references[0][0].requires_grad_()
        prediction = self.predict()
        for index in range(2):
            individual = self.model(
                self.image[index : index + 1],
                self.times[index : index + 1],
                self.text[index : index + 1],
                self.mask[index : index + 1],
                reference_latents=[self.references[index]],
            ).sample
            torch.testing.assert_close(prediction[index : index + 1], individual)
        prediction.square().mean().backward()
        self.assertGreater(self.references[0][0].grad.abs().sum().item(), 0)
        self.assertGreater(self.model.layers[0].proj["q"].weight.grad.abs().sum().item(), 0)
        changed = copy.deepcopy(self.references)
        changed[0][0] = changed[0][0].detach() + 2
        altered = self.model(self.image, self.times, self.text, self.mask, reference_latents=changed).sample
        self.assertFalse(torch.allclose(altered[0], prediction[0]))
        torch.testing.assert_close(altered[1], prediction[1])

    def test_split_attention_matches_explicit_reference_isolation_mask(self):
        class MaskedProcessor(MOSSImageAttentionProcessor):
            def __call__(self, attn, q, k, v, rope, reference_start=None, reference_kv=None, reference_kv_out=None):
                shape = (*q.shape[:2], attn.heads, q.shape[-1] // attn.heads)
                q, k = _rotate(q.reshape(shape), rope), _rotate(k.reshape(shape), rope)
                v = v.reshape(shape)
                allowed = torch.ones(q.shape[1], q.shape[1], device=q.device, dtype=torch.bool)
                if reference_start is not None:
                    allowed[reference_start:, :reference_start] = False
                return (
                    F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=allowed
                    )
                    .transpose(1, 2)
                    .flatten(2)
                )

        expected = self.predict()
        model = tiny_transformer()
        model.set_attn_processor(MaskedProcessor())
        torch.testing.assert_close(self.predict(model), expected, atol=1e-6, rtol=1e-5)

    def test_reference_keys_and_values_are_step_invariant(self):
        for interval_type in (None, "t-r"):
            with self.subTest(interval_type=interval_type), torch.no_grad():
                model = tiny_transformer(enable_time_sign_embed=True)
                model.time_sign_embed.weight[1].fill_(0.5)
                extra = {}
                if interval_type:
                    model.enable_flowmap_time_conditioning(0.25, interval_type)
                    extra["r_timestep"] = torch.tensor([100.0, 50.0])
                first, second = [{}, {}], [{}, {}]
                self.predict(model, _reference_cache=first, timestep_sign=torch.ones(2), **extra)
                model(
                    -self.image,
                    self.times / 2,
                    -self.text,
                    self.mask,
                    reference_latents=self.references,
                    _reference_cache=second,
                    timestep_sign=-torch.ones(2),
                    **extra,
                )
                for a, b in zip(first, second):
                    self.assertEqual(set(a), {0, 1, 2})
                    for index in a:
                        torch.testing.assert_close(a[index], b[index], atol=1e-6, rtol=1e-5)

    def test_cached_sampling_matches_uncached_cfg_and_changing_skip_windows(self):
        pipeline = MOSSImagePipeline(self.model)
        pipeline.set_progress_bar_config(disable=True)
        inputs = dict(
            prompt_embeds=self.text,
            prompt_attention_mask=self.mask,
            negative_prompt_embeds=-self.text,
            negative_prompt_attention_mask=self.mask,
            reference_latents=self.references,
            height=32,
            width=48,
            latents=self.image,
            num_inference_steps=4,
            guidance_scale=3.0,
            uncond_skip_layers=[1],
            slg_layers=[2],
            slg_scale=0.3,
            slg_start=0.2,
            slg_stop=0.7,
            num_images_per_prompt=1,
            output_type="latent",
        )
        eager = pipeline(**inputs, cache_reference_kv=False).images
        cached = pipeline(**inputs).images
        torch.testing.assert_close(cached, eager, atol=2e-6, rtol=2e-5)
        inputs.update(num_images_per_prompt=2, latents=self.image.repeat_interleave(2, 0), use_cfg_zero_star=True)
        torch.testing.assert_close(
            pipeline(**inputs).images, pipeline(**inputs, cache_reference_kv=False).images, atol=2e-6, rtol=2e-5
        )

    def test_cached_steps_skip_reference_projection(self):
        lengths = []
        handle = self.model.video_embedder.register_forward_pre_hook(lambda _module, args: lengths.append(args[0].shape[1]))
        with torch.no_grad():
            cache = [{}, {}]
            first = self.predict(_reference_cache=cache)
            self.assertEqual(len(lengths), 4)
            lengths.clear()
            second = self.predict(_reference_cache=cache)
            self.assertEqual(lengths, [6, 6])
            torch.testing.assert_close(first, second)
        handle.remove()

    def test_reference_rope_uses_image_index_and_centered_grid(self):
        baseline = self.model.rope(2, 3, 5, torch.device("cpu"))
        rope = self.model.rope(2, 3, 5, torch.device("cpu"), [(1, 2), (1, 1)])
        torch.testing.assert_close(rope[:, :11], baseline)
        coordinates = torch.tensor([[1, 5, 0, -0.5], [1, 5, 0, 0.5], [2, 5, 0, 0]])
        phase = coordinates[..., None] * self.model.rope.bands
        expected = torch.cat((phase.sin(), phase.cos()), dim=-2).flatten(-2)
        torch.testing.assert_close(rope[0, 11:], expected)

    def test_checkpointing_and_attention_offload_preserve_edit_gradients(self):
        expected = self.predict()
        expected.square().mean().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in self.model.named_parameters()}
        for backend in ("torch", "torch-ffn", "unsloth", "unsloth-ffn"):
            for interval in (1, 2):
                with self.subTest(backend=backend, interval=interval):
                    model = tiny_transformer()
                    model.enable_gradient_checkpointing()
                    model.set_gradient_checkpointing_backend(backend)
                    model.set_gradient_checkpointing_interval(interval)
                    model.set_gradient_checkpointing_offload_attention(True)
                    actual = self.predict(model)
                    torch.testing.assert_close(actual, expected)
                    actual.square().mean().backward()
                    for name, parameter in model.named_parameters():
                        torch.testing.assert_close(parameter.grad, gradients[name], atol=1e-6, rtol=1e-5)

    def test_tokenwise_self_flow_and_regularizers_capture_only_target_tokens(self):
        states = HiddenStateBuffer(capture_layers={0, 2})
        result = self.model(
            self.image,
            self.times[:, None].expand(-1, 6),
            self.text,
            self.mask,
            reference_latents=self.references,
            hidden_states_buffer=states,
        ).sample
        torch.testing.assert_close(result, self.predict(), atol=1e-6, rtol=1e-5)
        self.assertEqual(states["layer_0"].shape, (2, 6, 32))
        (result.square().mean() + states["layer_2"].square().mean()).backward()
        self.assertTrue(torch.isfinite(self.model.video_embedder.weight.grad).all())

    def test_edit_lora_training_and_pipeline_reload(self):
        from peft import LoraConfig
        from peft.utils import get_peft_model_state_dict

        restored = tiny_transformer()
        self.model.add_adapter(LoraConfig(r=2, lora_alpha=2, target_modules=MOSSImage.DEFAULT_LORA_TARGET))
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad], lr=0.01
        )
        before = self.predict().detach()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (self.predict() - 0.5).square().mean()
            loss.backward()
            optimizer.step()
        after = self.predict().detach()
        self.assertFalse(torch.allclose(before, after))
        with tempfile.TemporaryDirectory() as directory:
            MOSSImagePipeline.save_lora_weights(directory, transformer_lora_layers=get_peft_model_state_dict(self.model))
            MOSSImagePipeline(restored).load_lora_weights(directory)
        torch.testing.assert_close(self.predict(restored), after)

    def test_flavours_require_reference_datasets_and_distinct_caption_caches(self):
        for flavour in ("sft-edit", "base-edit"):
            model = self.foundation(flavour)
            self.assertTrue(model.requires_conditioning_dataset())
            self.assertTrue(model.requires_conditioning_latents())
            self.assertTrue(model.requires_text_embed_image_context())
            self.assertFalse(model.should_precompute_dropout_caption())
            self.assertEqual(model.text_embed_cache_key(), TextEmbedCacheKey.DATASET_AND_FILENAME)
            self.assertIn("moss-image-edit", model.text_embed_cache_namespace())
            first = {"image_paths": ["reference/a.png"], "data_backend_ids": ["refs"]}
            second = {"image_paths": ["reference/b.png"], "data_backend_ids": ["refs"]}
            self.assertNotEqual(model.text_embed_cache_context_key(first), model.text_embed_cache_context_key(second))
            self.assertIsNotNone(model.text_embed_cache_context_key(first, prompt=""))
            from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache

            cache = object.__new__(TextEmbeddingCache)
            cache.model, cache.model_type, cache.key_type = model, "moss_image", model.text_embed_cache_key()
            cache._requires_path_based_keys = True
            self.assertNotEqual(
                cache.create_hash("target:a", prompt="", metadata=first),
                cache.create_hash("target:a", prompt="edit", metadata=first),
            )
        self.assertFalse(self.foundation("sft").requires_conditioning_dataset())

    def test_batch_uses_cached_reference_grids_and_rejects_missing_references(self):
        model = self.foundation()
        batches = [[self.references[0][0], self.references[1][0]]]
        batch = {"latents": self.image, "conditioning_latents": batches}
        with patch.object(ImageModelFoundation, "prepare_batch_conditions", side_effect=lambda batch, state: batch):
            actual = model.prepare_batch_conditions(batch, {})
            model.model = self.model
            model._new_hidden_state_buffer = lambda: None
            model._get_flowmap_r_timestep_forward_kwargs = lambda batch: {}
            actual.update(
                noisy_latents=self.image, timesteps=self.times, prompt_embeds=self.text, encoder_attention_mask=self.mask
            )
            result = model.model_predict(actual)["model_prediction"]
            expected = self.model(
                self.image, self.times, self.text, self.mask, reference_latents=[[value] for value in batches[0]]
            ).sample
            torch.testing.assert_close(result, expected)
            from simpletuner.helpers.distillation.anyflow.distiller import AnyFlowDistiller

            distiller = object.__new__(AnyFlowDistiller)
            sliced = distiller._slice_batch(actual, 1)
            self.assertEqual(sliced["reference_inputs"]["tokens"].shape[0], 1)
            torch.testing.assert_close(model.model_predict(sliced)["model_prediction"], expected[:1])
            with self.assertRaisesRegex(ValueError, "cached reference"):
                model.prepare_batch_conditions({"latents": self.image}, {})

    def test_reference_input_validation_and_cache_requires_inference(self):
        for references in ([[]], [[torch.zeros(7, 2, 2)], []], [[torch.zeros(8, 0, 2)], []]):
            with self.subTest(references=references), self.assertRaises(ValueError):
                self.model(self.image, self.times, self.text, self.mask, reference_latents=references)
        with self.assertRaisesRegex(ValueError, "inference mode"):
            self.predict(_reference_cache=[{}, {}])

    def test_reference_image_batches_and_aspects_are_preserved(self):
        pipeline = MOSSImagePipeline(self.model)
        wide, tall = Image.new("RGB", (64, 32)), Image.new("RGB", (32, 64))
        images = pipeline._reference_images([wide, tall], 2)
        self.assertEqual([[value.size for value in sample] for sample in images], [[(64, 32), (32, 64)]] * 2)
        individual = pipeline._reference_images([[wide], [tall]], 2)
        self.assertEqual([sample[0].size for sample in individual], [(64, 32), (32, 64)])

    def test_image_pipeline_matches_explicit_vae_reference_latents(self):
        from diffusers import AutoencoderKLFlux2

        vae = AutoencoderKLFlux2(
            block_out_channels=(8, 8, 8, 8),
            decoder_block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            latent_channels=2,
            norm_num_groups=4,
        ).eval()
        pipeline = MOSSImagePipeline(self.model, vae=vae)
        pipeline.set_progress_bar_config(disable=True)
        references = [Image.new("RGB", (64, 32), "blue"), Image.new("RGB", (32, 64), "red")]
        with torch.no_grad():
            grids = pipeline._encode_reference_latents([references], 32, 32, torch.device("cpu"))
        self.assertEqual([value.shape for value in grids[0]], [(8, 1, 3), (8, 3, 1)])
        inputs = dict(
            prompt_embeds=self.text[:1],
            prompt_attention_mask=self.mask[:1],
            height=32,
            width=32,
            latents=self.image[:1, :, :2, :2],
            num_inference_steps=2,
            guidance_scale=1.0,
            uncond_skip_layers=[],
            output_type="latent",
        )
        expected = pipeline(**inputs, reference_latents=grids).images
        actual = pipeline(**inputs, image=references).images
        torch.testing.assert_close(actual, expected)

    def test_vision_conditioning_uses_real_qwen_model_and_preserves_image_tokens(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessor, Qwen3_5Config, Qwen3_5Model

        from simpletuner.helpers.models.moss_image.text import encode_prompts

        vocabulary = {"[UNK]": 0, "<|vision_start|>": 1, "<|image_pad|>": 2, "<|vision_end|>": 3, "edit": 4}
        backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            additional_special_tokens=["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"],
            chat_template="{{ messages[0]['content'] }}",
        )
        config = Qwen3_5Config(
            image_token_id=2,
            vision_start_token_id=1,
            vision_end_token_id=3,
            video_token_id=5,
            text_config=dict(
                vocab_size=16,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=16,
                layer_types=["linear_attention", "full_attention"],
                linear_key_head_dim=16,
                linear_value_head_dim=16,
                linear_num_key_heads=2,
                linear_num_value_heads=2,
                partial_rotary_factor=0.5,
                rope_parameters=dict(
                    rope_type="default",
                    rope_theta=10000,
                    partial_rotary_factor=0.5,
                    mrope_section=[1, 1, 2],
                    mrope_interleaved=True,
                ),
            ),
            vision_config=dict(
                hidden_size=32,
                intermediate_size=64,
                out_hidden_size=32,
                depth=1,
                num_heads=4,
                num_position_embeddings=64,
                patch_size=16,
                spatial_merge_size=2,
                temporal_patch_size=2,
            ),
        )
        encoder = Qwen3_5Model(config).eval()
        processor = Qwen2VLImageProcessor(
            size={"shortest_edge": 1024, "longest_edge": 384 * 384},
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        )
        references = [[Image.new("RGB", (32, 32), "blue")], [Image.new("RGB", (32, 64), "red")]]
        embeds, masks = encode_prompts(
            tokenizer, encoder, ["edit", ""], "cpu", reference_images=references, image_processor=processor
        )
        self.assertTrue(torch.isfinite(embeds).all())
        self.assertGreater(masks[0].sum(), 1)
        self.assertGreater(masks[1].sum(), 1)
        single, _ = encode_prompts(
            tokenizer, encoder, [""], "cpu", reference_images=references[1:], image_processor=processor
        )
        torch.testing.assert_close(embeds[1, masks[1]], single[0])
        with self.assertRaisesRegex(ValueError, "exceed tokenizer_max_length"):
            encode_prompts(
                tokenizer, encoder, ["edit"], "cpu", max_length=2, reference_images=references[:1], image_processor=processor
            )

    def test_mps_edit_forward_backward_and_cache(self):
        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        model = self.model.to("mps")
        arguments = (self.image.to("mps"), self.times.to("mps"), self.text.to("mps"), self.mask.to("mps"))
        references = [[value.to("mps") for value in sample] for sample in self.references]
        result = model(*arguments, reference_latents=references).sample
        result.square().mean().backward()
        self.assertTrue(torch.isfinite(model.video_embedder.weight.grad).all())
        with torch.no_grad():
            cache = [{}, {}]
            initial = model(*arguments, reference_latents=references, _reference_cache=cache).sample
            reused = model(*arguments, reference_latents=references, _reference_cache=cache).sample
            torch.testing.assert_close(initial, reused, atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
