import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


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


if __name__ == "__main__":
    unittest.main()
