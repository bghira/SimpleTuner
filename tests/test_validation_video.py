import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import torch
from PIL import Image


class TestValidationVideoMux(unittest.TestCase):
    def test_mux_audio_uses_temp_mp4_extension(self):
        from simpletuner.helpers.training import validation_video

        video_path = "/tmp/validation_video.mp4"

        with (
            patch(
                "simpletuner.helpers.training.validation_video.validation_audio._tensor_to_wav_buffer",
                return_value=BytesIO(b"data"),
            ),
            patch("simpletuner.helpers.training.validation_video.shutil.which", return_value="/usr/bin/ffmpeg"),
            patch("simpletuner.helpers.training.validation_video.subprocess.run") as mock_run,
            patch("simpletuner.helpers.training.validation_video.os.replace") as mock_replace,
            patch("simpletuner.helpers.training.validation_video.os.path.exists", return_value=False),
        ):
            mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

            validation_video._mux_audio_into_video(video_path, MagicMock(), 16000)

        output_path = mock_run.call_args.args[0][-1]
        self.assertTrue(output_path.endswith(".tmp.mp4"))
        mock_replace.assert_called_once_with(output_path, video_path)

    def test_mux_audio_uses_imageio_ffmpeg_when_not_on_path(self):
        from simpletuner.helpers.training import validation_video

        video_path = "/tmp/validation_video.mp4"

        with (
            patch(
                "simpletuner.helpers.training.validation_video.validation_audio._tensor_to_wav_buffer",
                return_value=BytesIO(b"data"),
            ),
            patch("simpletuner.helpers.training.validation_video.shutil.which", return_value=None),
            patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="/venv/bin/imageio-ffmpeg"),
            patch("simpletuner.helpers.training.validation_video.subprocess.run") as mock_run,
            patch("simpletuner.helpers.training.validation_video.os.replace"),
            patch("simpletuner.helpers.training.validation_video.os.path.exists", return_value=False),
        ):
            mock_run.return_value = SimpleNamespace(returncode=0, stderr=b"")

            validation_video._mux_audio_into_video(video_path, MagicMock(), 16000)

        self.assertEqual(mock_run.call_args.args[0][0], "/venv/bin/imageio-ffmpeg")


class TestValidationVideoTrackerLogging(unittest.TestCase):
    def _accelerator(self):
        tensorboard = SimpleNamespace(
            name="tensorboard",
            writer=SimpleNamespace(add_video=MagicMock()),
            log_images=MagicMock(),
        )
        accelerator = SimpleNamespace(trackers=[SimpleNamespace(name="tensorboard")])
        accelerator.get_tracker = MagicMock(return_value=tensorboard)
        return accelerator, tensorboard

    @patch("simpletuner.helpers.training.validation_video._tensorboard_video_supported", return_value=True)
    @patch("simpletuner.helpers.training.validation_video.StateTracker.get_global_step", return_value=123)
    def test_tensorboard_logs_5d_video_tensor_with_add_video(self, _global_step, _video_supported):
        from simpletuner.helpers.training import validation_video

        accelerator, tensorboard = self._accelerator()
        video = torch.zeros(1, 3, 4, 8, 8)

        validation_video.log_videos_to_trackers(
            accelerator,
            {"sample": [video]},
            [(8, 8)],
            SimpleNamespace(framerate=12),
        )

        tensorboard.writer.add_video.assert_called_once()
        tag, logged_video = tensorboard.writer.add_video.call_args.args[:2]
        self.assertEqual(tag, "sample - (8, 8)")
        self.assertEqual(logged_video.shape, torch.Size([1, 4, 3, 8, 8]))
        self.assertEqual(tensorboard.writer.add_video.call_args.kwargs, {"global_step": 123, "fps": 12})
        tensorboard.log_images.assert_not_called()

    @patch("simpletuner.helpers.training.validation_video._tensorboard_video_supported", return_value=True)
    @patch("simpletuner.helpers.training.validation_video.StateTracker.get_global_step", return_value=123)
    def test_tensorboard_logs_channel_last_5d_video_tensor_with_add_video(self, _global_step, _video_supported):
        from simpletuner.helpers.training import validation_video

        accelerator, tensorboard = self._accelerator()
        video = torch.zeros(1, 4, 8, 8, 3)

        validation_video.log_videos_to_trackers(
            accelerator,
            {"sample": [video]},
            [(8, 8)],
            SimpleNamespace(framerate=12),
        )

        logged_video = tensorboard.writer.add_video.call_args.args[1]
        self.assertEqual(logged_video.shape, torch.Size([1, 4, 3, 8, 8]))

    @patch("simpletuner.helpers.training.validation_video._tensorboard_video_supported", return_value=False)
    @patch("simpletuner.helpers.training.validation_video.StateTracker.get_global_step", return_value=123)
    def test_tensorboard_falls_back_to_first_frame_without_moviepy(self, _global_step, _video_supported):
        from simpletuner.helpers.training import validation_video

        accelerator, tensorboard = self._accelerator()
        video = torch.zeros(1, 4, 8, 8, 3)

        validation_video.log_videos_to_trackers(
            accelerator,
            {"sample": [video]},
            [(8, 8)],
            SimpleNamespace(framerate=12),
        )

        tensorboard.writer.add_video.assert_not_called()
        tensorboard.log_images.assert_called_once()
        payload = tensorboard.log_images.call_args.args[0]
        self.assertEqual(payload["sample - (8, 8)"].shape, (1, 3, 8, 8))

    @patch("simpletuner.helpers.training.validation_video.StateTracker.get_global_step", return_value=124)
    def test_tensorboard_falls_back_to_image_logging_for_static_image(self, _global_step):
        from simpletuner.helpers.training import validation_video

        accelerator, tensorboard = self._accelerator()
        image = Image.new("RGB", (4, 5))

        validation_video.log_videos_to_trackers(
            accelerator,
            {"sample": [image]},
            [(4, 5)],
            SimpleNamespace(framerate=None),
        )

        tensorboard.writer.add_video.assert_not_called()
        tensorboard.log_images.assert_called_once()
        payload = tensorboard.log_images.call_args.args[0]
        self.assertEqual(payload["sample - (4, 5)"].shape, (1, 3, 5, 4))
        self.assertEqual(tensorboard.log_images.call_args.kwargs, {"step": 124})

    def test_validation_routes_video_models_to_video_tracker_logger(self):
        from simpletuner.helpers.training import validation as validation_module

        class FakeVideoModel:
            def validation_audio_sample_rate(self):
                return None

        validation = validation_module.Validation.__new__(validation_module.Validation)
        validation.model = FakeVideoModel()
        validation.accelerator = MagicMock()
        validation.validation_resolutions = [(8, 8)]
        validation.config = SimpleNamespace()

        with (
            patch.object(validation_module, "VideoModelFoundation", FakeVideoModel),
            patch.object(validation_module.validation_video, "log_videos_to_trackers") as log_videos,
            patch.object(validation_module.validation_images_utils, "log_images_to_trackers") as log_images,
        ):
            validation._log_validations_to_trackers({"sample": [torch.zeros(1, 3, 2, 8, 8)]})

        log_videos.assert_called_once_with(
            validation.accelerator,
            {"sample": [ANY]},
            [(8, 8)],
            validation.config,
        )
        log_images.assert_not_called()


if __name__ == "__main__":
    unittest.main()
