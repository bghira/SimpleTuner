import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.local import LocalDataBackend


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required to generate a video fixture")
class S2VAudioHelperTests(unittest.TestCase):
    def test_checkout_tool_writes_sidecars_resolved_by_local_dataset(self):
        tool = Path(__file__).resolve().parents[1] / "scripts" / "generate_s2v_audio.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "video"
            video_root.mkdir()
            first = video_root / "first" / "clip.mp4"
            first.parent.mkdir()
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=s=16x16:d=0.2",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=duration=0.2",
                    "-c:v",
                    "mpeg4",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(first),
                ],
                check=True,
                capture_output=True,
            )
            flat = root / "flat"
            command = [sys.executable, str(tool), "--input-dir", str(video_root)]
            subprocess.run([*command, "--output-dir", str(flat)], check=True, capture_output=True)

            second = video_root / "second" / first.name
            second.parent.mkdir()
            shutil.copyfile(first, second)
            collision = root / "collision"
            rejected = subprocess.run(
                [*command, "--output-dir", str(collision), "--strip-audio"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("would both write clip.wav", rejected.stderr)
            self.assertFalse(collision.exists())
            self.assertEqual(first.read_bytes(), second.read_bytes())

            mirrored = root / "mirrored"
            subprocess.run(
                [*command, "--output-dir", str(mirrored), "--sidecar-match", "relative_path"],
                check=True,
                capture_output=True,
            )

            video = DatasetRuntime("video", LocalDataBackend(None, "video"), None, {"instance_data_dir": str(video_root)})
            for mode, audio_root, paths in (
                ("stem", flat, [(first, "clip.wav")]),
                ("relative_path", mirrored, [(first, "first/clip.wav"), (second, "second/clip.wav")]),
            ):
                with self.subTest(mode=mode):
                    audio = DatasetRuntime(
                        "audio",
                        LocalDataBackend(None, "audio"),
                        None,
                        {"instance_data_dir": str(audio_root), "audio": {"sidecar_match": mode}},
                    )
                    video.audio_datasets = (audio,)
                    for source, expected in paths:
                        self.assertEqual(video.resolve_s2v_audio(str(source)), (audio, str(audio_root / expected)))
                        with wave.open(str(audio_root / expected)) as wav:
                            self.assertEqual((wav.getframerate(), wav.getnchannels()), (16000, 1))
                            self.assertGreater(wav.getnframes(), 0)
