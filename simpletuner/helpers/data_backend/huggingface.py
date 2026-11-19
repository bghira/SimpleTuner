import hashlib
import logging
import os
import tempfile
from bisect import bisect_left
from functools import cached_property
from io import BytesIO
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

import numpy as np
import torch
from atomicwrites import atomic_write
from PIL import Image

try:
    from torchvision.io.video_reader import VideoReader
except Exception:  # pragma: no cover - torchvision optional
    VideoReader = None  # type: ignore[assignment]

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType, ensure_dataset_type
from simpletuner.helpers.image_manipulation.load import load_image, load_video
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger("HuggingfaceDatasetsBackend")


class HuggingfaceDatasetsBackend(BaseDataBackend):
    _CAP_PROP_POS_FRAMES = 1
    _CAP_PROP_FRAME_WIDTH = 3
    _CAP_PROP_FRAME_HEIGHT = 4
    _CAP_PROP_FPS = 5
    _CAP_PROP_FRAME_COUNT = 7

    def __init__(
        self,
        accelerator,
        id: str,
        dataset_name: str,
        dataset_config: Optional[str] = None,
        data_files: Optional[Any] = None,
        split: str = "train",
        revision: str = None,
        image_column: str = "image",
        video_column: str = "video",
        audio_column: str = "audio",
        cache_dir: Optional[str] = None,
        compress_cache: bool = False,
        streaming: bool = False,
        filter_func: Optional[callable] = None,
        num_proc: int = 16,
        composite_config: dict = {},
        dataset_type: Union[str, DatasetType] = DatasetType.IMAGE,
        auto_load: bool = False,
    ):
        self.id = id
        self.type = "huggingface"
        self.accelerator = accelerator
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.data_files = data_files
        self.dataset_type = ensure_dataset_type(dataset_type, default=DatasetType.IMAGE)
        if self.dataset_type is DatasetType.VIDEO:
            default_extension = "mp4"
        elif self.dataset_type is DatasetType.AUDIO:
            default_extension = "wav"
        else:
            default_extension = "jpg"
        self.file_extension = default_extension
        self.split = split
        self.revision = revision
        self.image_column = image_column
        self.video_column = video_column
        self.audio_column = audio_column
        self.cache_dir = cache_dir
        self.compress_cache = compress_cache
        self.streaming = streaming
        self.filter_func = filter_func
        self.num_proc = num_proc
        self.composite_config = composite_config
        self._auto_load = auto_load

        self._dataset = None
        self._source_shard_lengths = None
        self._dataset_loaded = False
        self._loading_dataset = False

        if should_log():
            logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
        else:
            logger.setLevel("ERROR")

        if self._auto_load:
            self._ensure_dataset_loaded()

    def _load_dataset(self):
        if self._loading_dataset:
            return
        self._loading_dataset = True
        try:
            from datasets import Features, Value, load_dataset, load_dataset_builder
        except ImportError:
            self._loading_dataset = False
            raise ImportError("Please install datasets: pip install datasets")

        dataset_features = None
        download_mode = "reuse_dataset_if_exists"
        if self.dataset_type is DatasetType.AUDIO and self.dataset_name == "audiofolder":
            dataset_features = Features({self.audio_column: Value("string")})
            download_mode = "force_redownload"

        # First, inspect the dataset structure
        logger.info(f"Inspecting dataset {self.dataset_name}")
        builder = load_dataset_builder(
            self.dataset_name,
            name=self.dataset_config,
            data_files=self.data_files,
            features=dataset_features,
            cache_dir=self.cache_dir,
        )
        logger.info(f"Dataset info: {builder.info}")
        try:
            available_splits = list(builder.info.splits.keys()) if builder.info.splits else []
        except Exception:
            available_splits = []
        logger.info(f"Available splits: {available_splits}")

        logger.info(f"Loading dataset {self.dataset_name} (split: {self.split})")

        # Load with explicit parameters to ensure all data is loaded
        try:
            self.dataset = load_dataset(
                self.dataset_name,
                name=self.dataset_config,
                data_files=self.data_files,
                split=self.split,
                revision=self.revision,
                cache_dir=self.cache_dir,
                features=dataset_features,
                streaming=self.streaming,
                download_mode=download_mode,
            )

            source_shard_lengths = None
            if not self.streaming:
                split_info = (self.dataset.info.splits or {}).get(self.split)
                lengths = split_info.shard_lengths if split_info is not None else None
                if lengths and len(lengths) > 1 and sum(lengths) == len(self.dataset):
                    source_shard_lengths = tuple(lengths)

            # Log the initial size
            if not self.streaming:
                logger.info(f"Loaded dataset with {len(self.dataset)} items")

                # Check if this looks like a subset
                if len(self.dataset) == 8200:
                    logger.warning("Dataset has exactly 8200 items - this might be a single shard!")

            self._configure_video_column()
            self._configure_audio_column()

            # Apply filter if provided
            if self.filter_func and not self.streaming:
                logger.info("Applying filter to dataset...")
                original_size = len(self.dataset)
                self.dataset = self.dataset.filter(
                    self.filter_func,
                    num_proc=self.num_proc,
                )
                logger.info(f"Dataset filtered from {original_size} to {len(self.dataset)} items")

            self._source_shard_lengths = source_shard_lengths
            self._dataset_loaded = True
        finally:
            if not self._dataset_loaded:
                self._dataset = None
            self._loading_dataset = False

    def _configure_video_column(self) -> None:
        if self.dataset_type is not DatasetType.VIDEO or self.streaming:
            return

        try:
            from datasets import Video
        except ImportError:
            logger.warning("datasets.Video not available; cannot disable video decoding.")
            return

        try:
            self.dataset = self.dataset.cast_column(self.video_column, Video(decode=False))
        except Exception as exc:
            logger.warning("Failed to cast video column '%s' to decode=False: %s", self.video_column, exc)

    def _configure_audio_column(self) -> None:
        if self.dataset_type is not DatasetType.AUDIO or self.streaming or self.dataset_name == "audiofolder":
            return

        try:
            from datasets import Audio
        except ImportError:
            logger.warning("datasets.Audio not available; cannot disable audio decoding.")
            return

        try:
            self.dataset = self.dataset.cast_column(self.audio_column, Audio(decode=False))
        except Exception as exc:
            logger.warning("Failed to cast audio column '%s' to decode=False: %s", self.audio_column, exc)

    @staticmethod
    def _coerce_to_bytes(payload: Any) -> Optional[bytes]:
        if payload is None:
            return None
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, memoryview):  # type: ignore[name-defined]
            return payload.tobytes()
        if isinstance(payload, np.ndarray):
            try:
                return payload.tobytes()
            except Exception:
                return None
        if hasattr(payload, "tobytes"):
            try:
                return payload.tobytes()
            except Exception:
                return None
        if isinstance(payload, list):
            try:
                return bytes(payload)
            except Exception:
                return None
        return None

    def _prepare_video_source(self, sample: Any) -> tuple[Optional[str], Optional[str]]:
        """Return a filesystem path to the video and an optional temp path for cleanup."""

        path_candidate: Optional[str] = None
        bytes_payload: Optional[bytes] = None

        if isinstance(sample, dict):
            raw_bytes = self._coerce_to_bytes(sample.get("bytes"))
            if raw_bytes:
                bytes_payload = raw_bytes
            path_val = sample.get("path")
            if isinstance(path_val, (str, Path)):
                path_candidate = str(path_val)
        elif isinstance(sample, (str, Path)):
            path_candidate = str(sample)
        elif isinstance(sample, (bytes, bytearray, memoryview)):
            bytes_payload = self._coerce_to_bytes(sample)

        if path_candidate:
            if os.path.isfile(path_candidate):
                return path_candidate, None
            try:
                from datasets.utils.file_utils import xopen

                with xopen(path_candidate, "rb") as file_obj:
                    bytes_payload = file_obj.read()
            except Exception as exc:
                logger.debug("Failed to open video path '%s' via xopen: %s", path_candidate, exc)

        if not bytes_payload:
            return None, None

        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(bytes_payload)
            temp_file.flush()
            temp_file.close()
            return temp_file.name, temp_file.name
        except Exception as exc:
            logger.error("Unable to create temporary video file for metadata extraction: %s", exc)
            return None, None

    def _metadata_from_video_reader(self, video: Any) -> Dict[str, Any]:
        if VideoReader is None or video is None:
            return {}
        metadata: Dict[str, Any] = {}
        try:
            md = video.get_metadata().get("video", {})
        except Exception as exc:
            logger.debug("Failed to read metadata from VideoReader: %s", exc)
            md = {}

        fps_values = md.get("fps") if isinstance(md, dict) else None
        if fps_values:
            try:
                metadata["fps"] = float(fps_values[0])
            except Exception:
                pass

        duration_values = md.get("duration") if isinstance(md, dict) else None
        if duration_values and "fps" in metadata:
            try:
                metadata["num_frames"] = int(round(duration_values[0] * metadata["fps"]))
            except Exception:
                pass

        try:
            iterator = iter(video)
            first_batch = next(iterator)
            frame_shape = first_batch["data"].shape if "data" in first_batch else None
        except StopIteration:
            frame_shape = None
        except Exception as exc:
            logger.debug("Failed to sample frame from VideoReader: %s", exc)
            frame_shape = None
        finally:
            try:
                video.seek(0)
            except Exception:
                pass

        if frame_shape is not None:
            if len(frame_shape) == 3:
                _, height, width = frame_shape
                metadata["original_size"] = (width, height)
            elif len(frame_shape) >= 2:
                metadata["original_size"] = (frame_shape[-1], frame_shape[-2])

        return metadata

    def _metadata_with_trainingsample(self, sample: Any) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}
        video_path, temp_path = self._prepare_video_source(sample)
        if not video_path:
            return metadata
        reader = None
        capture = None
        try:
            if VideoReader is not None:
                try:
                    reader = VideoReader(video_path, "video")
                    metadata.update(self._metadata_from_video_reader(reader))
                except Exception as exc:
                    logger.debug("Failed to extract metadata with torchvision VideoReader: %s", exc)

            import trainingsample as tsr

            capture = tsr.PyVideoCapture(video_path)
            if not capture.is_opened():
                return metadata

            try:
                fps = capture.get(self._CAP_PROP_FPS)
                if fps and fps > 0:
                    metadata["fps"] = float(fps)
            except Exception:
                pass

            try:
                frame_count = capture.get(self._CAP_PROP_FRAME_COUNT)
                if frame_count and frame_count > 0:
                    metadata["num_frames"] = int(frame_count)
            except Exception:
                pass

            width = height = None
            try:
                width = capture.get(self._CAP_PROP_FRAME_WIDTH)
                height = capture.get(self._CAP_PROP_FRAME_HEIGHT)
            except Exception:
                width = height = None

            if width and height and width > 0 and height > 0:
                metadata["original_size"] = (int(width), int(height))
            else:
                try:
                    success, frame = capture.read()
                    if success and frame is not None:
                        metadata["original_size"] = (frame.shape[1], frame.shape[0])
                except Exception:
                    pass
        except Exception as exc:
            logger.debug("Failed to extract metadata with trainingsample: %s", exc)
        finally:
            if capture is not None:
                try:
                    capture.release()
                except Exception:
                    pass
            if reader is not None:
                try:
                    reader.close()
                except Exception:
                    pass
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

        return metadata

    def _extract_video_sample_metadata(self, sample: Any) -> Dict[str, Any]:
        if VideoReader is not None and isinstance(sample, VideoReader):
            return self._metadata_from_video_reader(sample)
        return self._metadata_with_trainingsample(sample)

    def _ensure_dataset_loaded(self) -> None:
        if self._dataset_loaded:
            return
        self._load_dataset()

    @property
    def dataset(self):
        self._ensure_dataset_loaded()
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self._source_shard_lengths = None
        self.__dict__.pop("preparation_shards", None)
        self._dataset_loaded = value is not None

    @cached_property
    def preparation_shards(self):
        """Logical row ranges aligned with the published Arrow shards of one full split."""
        dataset = self.dataset
        if self._source_shard_lengths is None:
            return None
        boundaries = accumulate(self._source_shard_lengths, initial=0)
        # Dataset.filter preserves physical row order. Its Arrow index maps logical rows to physical rows.
        if dataset._indices is not None:
            positions = dataset._indices.column("indices")
            boundaries = (
                bisect_left(range(len(dataset)), boundary, key=lambda i: positions[i].as_py()) for boundary in boundaries
            )
        return tuple(range(start, end) for start, end in pairwise(boundaries))

    def _get_index_from_path(self, filepath: str) -> Optional[int]:
        basename = os.path.basename(filepath)
        try:
            return int(os.path.splitext(basename)[0])
        except ValueError:
            logger.error(f"Could not extract index from path: {filepath}")
            raise

    def get_instance_representation(self) -> dict:
        return {
            "backend_type": "huggingface",
            "id": self.id,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "data_files": self.data_files,
            "dataset_type": self.dataset_type.value,
            "split": self.split,
            "revision": self.revision,
            "image_column": self.image_column,
            "video_column": self.video_column,
            "audio_column": self.audio_column,
            "cache_dir": self.cache_dir,
            "compress_cache": self.compress_cache,
            "streaming": self.streaming,
            # Note: filter_func is not serializable
            "filter_func_name": self.filter_func.__name__ if self.filter_func else None,
            "num_proc": self.num_proc,
            "composite_config": self.composite_config,
        }

    @staticmethod
    def from_instance_representation(
        representation: dict,
    ) -> "HuggingfaceDatasetsBackend":
        if representation.get("backend_type") != "huggingface":
            raise ValueError(f"Expected backend_type 'huggingface', got {representation.get('backend_type')}")

        # filter_func cannot be serialized/deserialized automatically
        filter_func = None
        if representation.get("filter_func_name"):
            import logging

            logger = logging.getLogger("HuggingfaceDatasetsBackend")
            logger.warning(
                f"Filter function '{representation['filter_func_name']}' cannot be automatically restored in subprocess"
            )

        return HuggingfaceDatasetsBackend(
            accelerator=None,  # Will be set by subprocess if needed
            id=representation["id"],
            dataset_name=representation["dataset_name"],
            dataset_config=representation.get("dataset_config"),
            data_files=representation.get("data_files"),
            split=representation.get("split", "train"),
            revision=representation.get("revision"),
            image_column=representation.get("image_column", "image"),
            video_column=representation.get("video_column", "video"),
            audio_column=representation.get("audio_column", "audio"),
            cache_dir=representation.get("cache_dir"),
            compress_cache=representation.get("compress_cache", False),
            streaming=representation.get("streaming", False),
            filter_func=filter_func,  # Would need special handling
            num_proc=representation.get("num_proc", 16),
            composite_config=representation.get("composite_config", {}),
            dataset_type=representation.get("dataset_type", DatasetType.IMAGE),
            auto_load=False,
        )

    _CACHE_EXTENSIONS = (".json", ".pt", ".msgpack", ".safetensors")

    def _scoped_cache_file(self, filepath):
        path = Path(filepath)
        if path.suffix == ".json":
            folder = "huggingface_metadata"
        elif path.suffix in {".pt", ".safetensors"}:
            folder = "vae"
        else:
            folder = "cache"
        return Path(self.cache_dir) / folder / self.id / path.name

    def _existing_cache_file(self, filepath):
        path = Path(filepath)
        if not path.exists() and self.cache_dir:
            return self._scoped_cache_file(path)
        return path

    def open_read_stream(self, identifier):
        return self._existing_cache_file(identifier).open("rb")

    def read(self, location, as_byteIO: bool = False):
        if isinstance(location, Path):
            location = str(location)

        if location.endswith(self._CACHE_EXTENSIONS):
            try:
                with self.open_read_stream(location) as stream:
                    data = stream.read()
            except FileNotFoundError:
                logger.error("Cache file not found: %s", location)
                return None
            return BytesIO(data) if as_byteIO else data

        # Handle virtual dataset files (the existing logic)
        self._ensure_dataset_loaded()
        index = self._get_index_from_path(location)
        if index is None:
            logger.error(f"Invalid path: {location}")
            return None

        try:
            # Get the item from dataset
            item = self.dataset[index]
            if self.dataset_type is DatasetType.VIDEO:
                column = self.video_column
            elif self.dataset_type is DatasetType.AUDIO:
                column = self.audio_column
            else:
                column = self.image_column
            sample = item.get(column)

            if sample is None:
                logger.error(f"No {self.dataset_type} found in column '{column}' for index {index}")
                return None

            # New HF audio types (torchcodec AudioDecoder) need explicit decode.
            if self.dataset_type is DatasetType.AUDIO:
                # Prefer direct decode() on torchcodec AudioDecoder objects
                if hasattr(sample, "decode"):
                    try:
                        decoded = sample.decode()
                        sample_rate = getattr(sample, "sampling_rate", getattr(sample, "sample_rate", None))
                        if isinstance(decoded, dict):
                            decoded_array = decoded.get("array") or decoded.get("audio") or decoded.get("data")
                            if decoded_array is None:
                                decoded_array = decoded
                            if torch.is_tensor(decoded_array):
                                decoded_array = decoded_array.detach().cpu().numpy()
                            sample = {"array": decoded_array, "sampling_rate": sample_rate or decoded.get("sampling_rate")}
                        elif isinstance(decoded, (list, tuple)):
                            wave = decoded[0]
                            if torch.is_tensor(wave):
                                wave = wave.detach().cpu().numpy()
                            sr = decoded[1] if len(decoded) > 1 else sample_rate
                            sample = {"array": wave, "sampling_rate": sr}
                        elif torch.is_tensor(decoded):
                            sample = {"array": decoded.detach().cpu().numpy(), "sampling_rate": sample_rate}
                        else:
                            sample = {"array": decoded, "sampling_rate": sample_rate}
                    except Exception as exc:
                        logger.error("Failed to decode audio sample for index %s: %s", index, exc)
                        return None
                elif hasattr(sample, "decode_example"):
                    try:
                        sample = sample.decode_example(None)
                    except Exception:
                        try:
                            sample = sample.decode_example({"path": None, "bytes": None})
                        except Exception as exc:
                            logger.error("Failed to decode audio sample for index %s: %s", index, exc)
                            return None
                elif hasattr(sample, "get_all_samples"):
                    try:
                        decoded_samples = sample.get_all_samples()
                        sample_rate = getattr(decoded_samples, "sample_rate", None)
                        if not sample_rate:
                            metadata = getattr(sample, "metadata", None)
                            sample_rate = getattr(metadata, "sample_rate", None)
                        audio_data = getattr(decoded_samples, "data", None)
                        if audio_data is None and hasattr(decoded_samples, "samples"):
                            audio_data = getattr(decoded_samples, "samples", None)
                        if torch.is_tensor(audio_data):
                            audio_data = audio_data.detach().cpu().numpy()
                        sample = {"array": audio_data, "sampling_rate": sample_rate}
                        if sample["array"] is None:
                            raise ValueError("decoder did not return audio samples")
                        if sample["sampling_rate"] is None:
                            raise ValueError("decoder did not provide a sampling rate")
                    except Exception as exc:
                        logger.error("Failed to decode audio sample for index %s: %s", index, exc)
                        return None

            # Handle composite images if configured
            if (
                hasattr(self, "composite_config")
                and isinstance(self.composite_config, dict)
                and self.composite_config.get("enabled")
            ):
                image_count = self.composite_config.get("image_count", 1)
                select_index = self.composite_config.get("select_index", 0)

                if isinstance(sample, Image.Image):
                    width, height = sample.size
                    slice_width = width // image_count

                    # Calculate crop box for the selected index
                    left = select_index * slice_width
                    right = (select_index + 1) * slice_width

                    # Crop the image
                    sample = sample.crop((left, 0, right, height))

            # Handle different image types
            if isinstance(sample, Image.Image):
                # PIL Image - convert to bytes
                buffer = BytesIO()
                sample.save(buffer, format="PNG")
                data = buffer.getvalue()
            elif isinstance(sample, np.ndarray):
                # NumPy array - convert to PIL then bytes
                pil_image = Image.fromarray(sample)
                buffer = BytesIO()
                pil_image.save(buffer, format="PNG")
                data = buffer.getvalue()
            elif isinstance(sample, bytes):
                # Already bytes
                data = sample
            elif isinstance(sample, (str, Path)):
                try:
                    path = str(sample)
                    if os.path.isfile(path):
                        with open(path, "rb") as f:
                            data = f.read()
                    else:
                        from datasets.utils.file_utils import xopen

                        with xopen(path, "rb") as f:
                            data = f.read()
                except Exception as exc:
                    logger.error("Failed to read %s sample from path '%s': %s", self.dataset_type, sample, exc)
                    data = None
                if data is None:
                    return None
            elif VideoReader is not None and isinstance(sample, VideoReader):
                # VideoReader - encode all frames into a video file in memory
                import trainingsample as tsr

                frames = []
                for frame in sample:
                    # frame['data'] is a torch tensor (T, H, W, C)
                    frame_np = frame["data"].numpy()
                    # Convert from RGB to BGR for video encoding
                    frame_np = tsr.cvt_color_py(frame_np, 5)  # 5 = COLOR_RGB2BGR
                    frames.append(frame_np)

                if not frames:
                    logger.error("VideoReader contains no frames")
                    return None

                height, width, channels = frames[0].shape
                fourcc = tsr.fourcc_py("m", "p", "4", "v")
                temp_video = BytesIO()
                # OpenCV cannot write directly to BytesIO, so use a temp file
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpfile:
                    out = tsr.PyVideoWriter(
                        tmpfile.name,
                        fourcc,
                        sample.get_metadata()["video"]["fps"][0],
                        (width, height),
                    )
                    for frame in frames:
                        out.write(frame)
                    out.release()
                    tmpfile.seek(0)
                    data = tmpfile.read()
                # Now data contains the video bytes
            elif isinstance(sample, dict):
                bytes_data = sample.get("bytes")
                path = sample.get("path")
                array = sample.get("array")
                sample_rate = sample.get("sampling_rate")
                data = None

                if array is not None and sample_rate:
                    try:
                        import numpy as _np
                        import soundfile as sf  # type: ignore
                    except Exception:
                        import numpy as _np  # type: ignore

                        sf = None
                    try:
                        waveform = _np.asarray(array)
                        if waveform.ndim == 1:
                            waveform = waveform[None, :]
                        waveform = waveform.T  # (samples, channels)
                        buffer = BytesIO()
                        if sf is not None:
                            sf.write(buffer, waveform, int(sample_rate), format="WAV")
                        else:
                            import torchaudio

                            tensor = torch.from_numpy(waveform.T)
                            torchaudio.save(buffer, tensor, int(sample_rate), format="wav")
                        buffer.seek(0)
                        data = buffer.read()
                    except Exception as exc:
                        logger.error("Failed to convert audio array to bytes for index %s: %s", index, exc)
                        data = None

                if data is None and bytes_data is not None:
                    try:
                        data = bytes(bytes_data)
                    except Exception:
                        logger.error(
                            "Unable to convert '%s' bytes payload to raw bytes.",
                            self.video_column if self.dataset_type is DatasetType.VIDEO else self.audio_column,
                        )
                        data = None
                if data is None and path:
                    try:
                        if os.path.isfile(path):
                            with open(path, "rb") as f:
                                data = f.read()
                        else:
                            from datasets.utils.file_utils import xopen

                            with xopen(path, "rb") as f:
                                data = f.read()
                    except Exception as exc:
                        logger.error("Failed to read %s sample from path '%s': %s", self.dataset_type, path, exc)
                        data = None

                if data is None:
                    logger.error(
                        "Dataset sample in column '%s' missing usable data (keys=%s)",
                        self.video_column if self.dataset_type is DatasetType.VIDEO else self.audio_column,
                        list(sample.keys()),
                    )
                    return None
            else:
                logger.error(f"Unsupported image type: {type(sample)}")
                return None

            if as_byteIO:
                return BytesIO(data)
            return data

        except Exception as e:
            logger.error(f"Error reading from dataset at index {index}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            raise

    def cache_write_path(self, filepath):
        path = Path(filepath)
        if path.is_absolute() and path.parent.is_dir():
            return path
        if not self.cache_dir:
            raise ValueError(f"Cannot write cache file {filepath} - no cache_dir configured for HuggingFace backend")
        return self._scoped_cache_file(path)

    def write(self, filepath: Union[str, Path], data: Any) -> None:
        if isinstance(filepath, Path):
            filepath = str(filepath)

        if filepath.endswith(self._CACHE_EXTENSIONS):
            cache_path = self.cache_write_path(filepath)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            if filepath.endswith(".json"):
                if isinstance(data, (dict, list)):
                    import json

                    data = json.dumps(data)
                elif not isinstance(data, str):
                    data = str(data)

                with atomic_write(cache_path, mode="w", encoding="utf-8", overwrite=True) as f:
                    f.write(data)
            else:
                # Binary data
                with atomic_write(cache_path, mode="wb", overwrite=True) as f:
                    if isinstance(data, bytes):
                        f.write(data)
                    elif hasattr(data, "save"):
                        # For torch tensors or similar objects with save method
                        data.save(f)
                    else:
                        # Fallback to torch.save
                        import torch

                        torch.save(data, f)

            # cache file written
        else:
            logger.warning(f"Write operations are only supported for cache files, not {filepath}")
            raise NotImplementedError("Hugging Face datasets are read-only except for cache files")

    def exists(self, filepath):
        if isinstance(filepath, Path):
            filepath = str(filepath)

        if filepath.endswith(self._CACHE_EXTENSIONS):
            return self._existing_cache_file(filepath).exists()

        # For virtual files, check index
        self._ensure_dataset_loaded()
        index = self._get_index_from_path(filepath)
        if index is None:
            return False

        if not self.streaming:
            return 0 <= index < len(self.dataset)
        else:
            # For streaming, we can't easily check bounds
            return True

    def delete(self, filepath):
        logger.warning("Delete operations are not supported for Hugging Face datasets")
        raise NotImplementedError("Hugging Face datasets are read-only")

    def open_file(self, filepath, mode):
        if "w" in mode:
            raise NotImplementedError("Write operations are not supported")
        return BytesIO(self.read(filepath))

    def list_files(self, file_extensions: list = None, instance_data_dir: str = None) -> list:
        if instance_data_dir:
            # List files from the real directory
            instance_data_dir = str(instance_data_dir)
            if not os.path.exists(instance_data_dir):
                logger.warning(f"Directory does not exist: {instance_data_dir}")
                return []
            result = []
            for root, dirs, files in os.walk(instance_data_dir):
                filtered_files = []
                for f in files:
                    if file_extensions:
                        ext = os.path.splitext(f)[1].lower().strip(".")
                        if ext not in file_extensions:
                            continue
                    filtered_files.append(os.path.join(root, f))
                result.append((root, dirs, filtered_files))
            return result

        return [] if self.streaming else [("", [], list(self.iter_file_keys(file_extensions)))]

    def iter_file_keys(self, file_extensions=None, instance_data_dir=None):
        if instance_data_dir:
            yield from super().iter_file_keys(file_extensions, instance_data_dir)
        elif not self.streaming and (not file_extensions or self.file_extension in file_extensions):
            for index in range(len(self.dataset)):
                yield f"{index}.{self.file_extension}"

    def get_abs_path(self, sample_path: str) -> str:
        # for virtual paths, just return if it exists
        if self.exists(sample_path):
            return sample_path
        return None

    def read_image(self, filepath: str, delete_problematic_images: bool = False):
        try:
            if self.dataset_type is DatasetType.VIDEO:
                index = self._get_index_from_path(filepath)
                if index is None:
                    logger.error("Unable to resolve dataset index for %s", filepath)
                    return None
                item = self.get_dataset_item(index)
                if item is None:
                    return None
                sample = item.get(self.video_column)
                if sample is None:
                    logger.error("Dataset item %s missing '%s' column", filepath, self.video_column)
                    return None
                video_path, temp_path = self._prepare_video_source(sample)
                if not video_path:
                    logger.error("Unable to prepare video source for %s", filepath)
                    return None
                try:
                    return load_video(video_path)
                finally:
                    if temp_path:
                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

            image_data = self.read(filepath, as_byteIO=True)
            if image_data is None:
                return None
            return load_image(image_data)
        except Exception as e:
            logger.error(f"Error opening image {filepath}: {e}")
            if delete_problematic_images:
                logger.warning("Cannot delete from HF dataset - skipping problematic image")
            return None

    def read_image_batch(self, filepaths: list, delete_problematic_images: bool = False) -> list:
        output_images = []
        available_keys = []

        for filepath in filepaths:
            image = self.read_image(filepath, delete_problematic_images)
            if image is not None:
                output_images.append(image)
                available_keys.append(filepath)
            else:
                logger.warning(f"Unable to load image '{filepath}', skipping.")

        return (available_keys, output_images)

    def save_state(self):
        pass

    def get_dataset_item(self, index: int):
        self._ensure_dataset_loaded()
        if not self.streaming and (index < 0 or index >= len(self.dataset)):
            logger.warning(f"Retrieving {index=} for {len(self.dataset)=}, returning None")
            return None
        return self.dataset[index]

    def __len__(self):
        if self.streaming:
            logger.warning("Cannot get length of streaming dataset")
            return 0
        return len(self.dataset)

    def write_batch(self, filepaths: list, data_list: list) -> None:
        if len(filepaths) != len(data_list):
            raise ValueError("Number of filepaths must match number of data items")

        for filepath, data in zip(filepaths, data_list):
            self.write(filepath, data)

    def torch_load(self, filename):
        if isinstance(filename, Path):
            filename = str(filename)

        # Try to read the file
        data = self.read(filename, as_byteIO=True)
        if data is None:
            raise FileNotFoundError(f"Could not find file: {filename}")

        import torch

        return torch.load(data)

    def torch_save(self, data, location: Union[str, Path, BytesIO]):
        if isinstance(location, BytesIO):
            import torch

            torch.save(data, location)
        else:
            self.write(location, data)

    def create_directory(self, directory_path):
        os.makedirs(directory_path, exist_ok=True)


def _sanitize_sample_for_json(sample: Any) -> Any:
    """Convert non-JSON-serializable objects to string representations."""
    import json

    if sample is None:
        return None

    if isinstance(sample, list):
        return [_sanitize_sample_for_json(item) for item in sample]

    if isinstance(sample, dict):
        result = {}
        for key, value in sample.items():
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                # Non-serializable: show type info instead
                result[key] = f"<{type(value).__name__}>"
        return result

    try:
        json.dumps(sample)
        return sample
    except (TypeError, ValueError):
        return f"<{type(sample).__name__}>"


def test_huggingface_dataset(
    dataset_name: str,
    dataset_config: Optional[str] = None,
    split: Optional[str] = None,
    revision: Optional[str] = None,
    streaming: bool = False,
    token: Optional[str] = None,
    sample_count: int = 1,
) -> Dict[str, Any]:
    """Load lightweight dataset metadata and sample rows for validation."""

    if not dataset_name:
        raise ValueError("dataset_name is required")

    try:
        from datasets import load_dataset, load_dataset_builder
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("datasets library is required to test Hugging Face connections") from exc

    split = split or "train"

    try:
        builder = load_dataset_builder(
            dataset_name,
            name=dataset_config,
            revision=revision,
            token=token,
        )
    except Exception as exc:
        raise ValueError(f"Failed to load dataset metadata: {exc}") from exc

    features = list(getattr(builder.info, "features", {}).keys())
    available_splits = list(getattr(builder.info, "splits", {}).keys())
    sample = None
    sample_via_streaming = streaming

    if sample_count > 0:
        try:
            dataset_stream = load_dataset(
                dataset_name,
                name=dataset_config,
                split=split,
                revision=revision,
                streaming=True,
                token=token,
            )
            iterator = iter(dataset_stream)
            try:
                sample = next(iterator)
            except StopIteration:
                sample = None
            else:
                if sample_count > 1:
                    import itertools

                    samples = [sample]
                    samples.extend(list(itertools.islice(iterator, max(sample_count - 1, 0))))
                    sample = samples
        except Exception:
            sample_via_streaming = False
            try:
                limited_split = f"{split}[:{max(sample_count, 1)}]"
                dataset_slice = load_dataset(
                    dataset_name,
                    name=dataset_config,
                    split=limited_split,
                    revision=revision,
                    streaming=False,
                    token=token,
                )
                if sample_count == 1:
                    sample = dataset_slice[0] if len(dataset_slice) else None
                else:
                    sample = [dataset_slice[i] for i in range(min(sample_count, len(dataset_slice)))]
            except Exception as exc:
                raise ValueError(f"Failed to fetch dataset sample: {exc}") from exc

    info = builder.info
    dataset_size = None
    if hasattr(info, "splits") and split in info.splits:
        dataset_size = info.splits[split].num_examples

    return {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "revision": revision,
        "features": features,
        "available_splits": available_splits,
        "description": getattr(info, "description", None),
        "sample": _sanitize_sample_for_json(sample),
        "streaming_used": sample_via_streaming,
        "estimated_num_examples": dataset_size,
    }
