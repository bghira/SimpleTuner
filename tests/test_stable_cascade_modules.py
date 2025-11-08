import unittest
import unittest.mock as mock

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environments without torch
    torch = None

STABLE_CASCADE_IMPORT_ERROR = False
try:
    from simpletuner.helpers.models.common import PipelineTypes
    from simpletuner.helpers.models.stable_cascade import DDPMWuerstchenScheduler, StableCascadeStageC
    from simpletuner.helpers.models.stable_cascade.autoencoder import StableCascadeStageCAutoencoder
except Exception:  # pragma: no cover - missing deps
    PipelineTypes = None
    DDPMWuerstchenScheduler = None
    StableCascadeStageC = None
    StableCascadeStageCAutoencoder = None
    STABLE_CASCADE_IMPORT_ERROR = True
try:
    from simpletuner.helpers.models.registry import ModelRegistry
except Exception:  # pragma: no cover - missing deps
    ModelRegistry = None
    STABLE_CASCADE_IMPORT_ERROR = True


@unittest.skipIf(
    torch is None or STABLE_CASCADE_IMPORT_ERROR,
    "Stable Cascade autoencoder requirements are unavailable",
)
class StableCascadeAutoencoderTests(unittest.TestCase):
    @mock.patch("simpletuner.helpers.models.stable_cascade.autoencoder.models")
    def test_forward_normalizes_inputs(self, mock_models):
        class DummyFeatures(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.last_input = None

            def forward(self, x):
                self.last_input = x.detach().clone()
                # Mimic EfficientNet feature output: (batch, 1280, H, W)
                return torch.ones(x.shape[0], 1280, x.shape[2], x.shape[3], dtype=x.dtype, device=x.device)

        dummy_features = DummyFeatures()
        dummy_model = mock.Mock()
        dummy_model.features = dummy_features
        mock_models.efficientnet_v2_s.return_value = dummy_model

        class _Weights:
            IMAGENET1K_V1 = "weights"

        mock_models.EfficientNet_V2_S_Weights = _Weights

        autoencoder = StableCascadeStageCAutoencoder(latent_channels=4, dtype=torch.float32)
        test_tensor = torch.tensor([[[[-1.0]], [[0.0]], [[1.0]]]])
        with torch.no_grad():
            latents = autoencoder(test_tensor)

        self.assertEqual(latents.shape, (1, 4, 1, 1))
        self.assertFalse(torch.isnan(latents).any())
        normalized = dummy_features.last_input
        self.assertIsNotNone(normalized)
        expected = torch.tensor(
            [
                [(0.0 - 0.485) / 0.229],
                [((0.5) - 0.456) / 0.224],
                [((1.0) - 0.406) / 0.225],
            ]
        ).view(1, 3, 1, 1)
        self.assertTrue(torch.allclose(normalized, expected, atol=1e-5))


@unittest.skipIf(
    torch is None or STABLE_CASCADE_IMPORT_ERROR,
    "Stable Cascade scheduler requirements are unavailable",
)
class StableCascadeSchedulerTests(unittest.TestCase):
    def test_scheduler_step_shapes(self):
        scheduler = DDPMWuerstchenScheduler()
        scheduler.set_timesteps(4)
        self.assertGreaterEqual(len(scheduler.timesteps), 4)

        latents = torch.randn(1, 4, 2, 2)
        noise = torch.randn_like(latents)
        timestep = scheduler.timesteps[:1].to(latents.device)
        step_output = scheduler.step(noise, timestep, latents)
        self.assertEqual(step_output.prev_sample.shape, latents.shape)


@unittest.skipIf(
    STABLE_CASCADE_IMPORT_ERROR,
    "Stable Cascade modules could not be imported",
)
class StableCascadeRegistryTests(unittest.TestCase):
    def test_model_registry_entry_exists(self):
        model_cls = ModelRegistry.get("stable_cascade")
        self.assertIs(model_cls, StableCascadeStageC)


@unittest.skipIf(
    STABLE_CASCADE_IMPORT_ERROR,
    "Stable Cascade modules could not be imported",
)
class StableCascadeStageCTests(unittest.TestCase):
    def _build_model(self, config):
        def _fake_init(self, cfg, accelerator):
            self.config = cfg
            self.accelerator = accelerator
            self.pipelines = {}
            self.model = None
            self.controlnet = None
            self.vae = None
            self.text_encoders = []
            self.tokenizers = []
            self.noise_schedule = mock.MagicMock()

        with mock.patch(
            "simpletuner.helpers.models.stable_cascade.model.ImageModelFoundation.__init__",
            _fake_init,
        ):
            return StableCascadeStageC(config, accelerator=None)

    def test_check_user_config_requires_fp32(self):
        config = mock.MagicMock()
        config.mixed_precision = "bf16"
        config.i_know_what_i_am_doing = False
        config.base_model_precision = "no_change"
        config.tokenizer_max_length = 128

        model = self._build_model(config)
        with self.assertRaises(ValueError):
            model.check_user_config()

    def test_check_user_config_overrides_token_length(self):
        config = mock.MagicMock()
        config.mixed_precision = "no"
        config.i_know_what_i_am_doing = False
        config.base_model_precision = "no_change"
        config.tokenizer_max_length = 128

        model = self._build_model(config)
        model.check_user_config()
        self.assertEqual(config.tokenizer_max_length, 77)

    @mock.patch(
        "simpletuner.helpers.models.stable_cascade.model.ensure_wuerstchen_scheduler",
        side_effect=lambda scheduler: scheduler,
    )
    @mock.patch("simpletuner.helpers.models.stable_cascade.model.StableCascadeCombinedPipeline.from_pretrained")
    def test_get_pipeline_uses_combined_wrapper(self, mock_from_pretrained, _mock_ensure):
        config = mock.MagicMock()
        config.model_flavour = "stage-c"
        config.pretrained_model_name_or_path = "prior"
        config.pretrained_vae_model_name_or_path = "prior"
        config.vae_path = "prior"
        config.stable_cascade_use_decoder_for_validation = True
        config.stable_cascade_decoder_model_name_or_path = "decoder-repo"
        config.stable_cascade_decoder_variant = None
        config.stable_cascade_decoder_subfolder = None
        config.stable_cascade_decoder_dtype = "fp16"
        config.weight_dtype = torch.float32
        config.validation_guidance = 3.0
        config.stable_cascade_validation_prior_num_inference_steps = 9
        config.stable_cascade_validation_prior_guidance_scale = 5.5
        config.stable_cascade_validation_decoder_guidance_scale = 2.0

        dummy_pipeline = mock.MagicMock()
        dummy_pipeline.prior_pipe = mock.MagicMock()
        dummy_pipeline.decoder_pipe = mock.MagicMock()
        dummy_pipeline.prior_scheduler = mock.MagicMock()
        mock_from_pretrained.return_value = dummy_pipeline

        model = self._build_model(config)
        prior_unet = mock.MagicMock()
        model.model = prior_unet
        model.text_encoders = [mock.MagicMock()]
        model.tokenizers = [mock.MagicMock()]
        model.unwrap_model = mock.MagicMock(return_value=prior_unet)

        pipeline = model.get_pipeline(load_base_model=False)
        self.assertIs(pipeline, dummy_pipeline)
        self.assertIs(dummy_pipeline.prior_pipe.prior, prior_unet)
        self.assertIs(model._combined_validation_pipeline, dummy_pipeline)

        kwargs = model.update_pipeline_call_kwargs({})
        self.assertEqual(kwargs["prior_num_inference_steps"], 9)
        self.assertEqual(kwargs["prior_guidance_scale"], 5.5)
        self.assertEqual(kwargs["decoder_guidance_scale"], 2.0)


if __name__ == "__main__":
    unittest.main()
