from typing import Any

from torch.utils.data import Dataset
from helpers.training.state_tracker import StateTracker
from helpers.multiaspect.image import MultiaspectImage
from helpers.image_manipulation.training_sample import TrainingSample
import logging
import os

logger = logging.getLogger("MultiAspectDataset")
from helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class MultiAspectDataset(Dataset):
    """
    A multi-aspect dataset requires special consideration and handling.
    This class implements bucketed data loading for precomputed text embeddings.
    This class does not do any image transforms, as those are handled by VAECache.
    """

    def __init__(
        self,
        id: str,
        datasets: list,
        print_names: bool = False,
        is_regularisation_data: bool = False,
        is_i2v_data: bool = False,
    ):
        self.id = id
        self.datasets = datasets
        self.print_names = print_names
        self.is_regularisation_data = is_regularisation_data
        self.is_i2v_data = is_i2v_data

    def __len__(self):
        # Sum the length of all data backends:
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, image_tuple: list[dict[str, Any] | TrainingSample]):
        output_data = {
            "training_samples": [],
            "conditioning_samples": [],
            "is_regularisation_data": self.is_regularisation_data,
            "is_i2v_data": self.is_i2v_data,
        }
        first_aspect_ratio = None
        for sample in image_tuple:
            # pick out the TrainingSamples, which represent conditioning samples
            if isinstance(sample, TrainingSample):
                image_metadata = sample.image_metadata
                output_data["conditioning_samples"].append(sample)
                continue

            image_metadata = sample
            if "target_size" in image_metadata:
                calculated_aspect_ratio = MultiaspectImage.calculate_image_aspect_ratio(
                    image_metadata["target_size"]
                )
                if first_aspect_ratio is None:
                    first_aspect_ratio = calculated_aspect_ratio
                elif first_aspect_ratio != calculated_aspect_ratio:
                    raise ValueError(
                        f"Aspect ratios must be the same for all images in a batch. Expected: {first_aspect_ratio}, got: {calculated_aspect_ratio}"
                    )

            if "deepfloyd" not in StateTracker.get_args().model_family and (
                image_metadata["original_size"] is None
                or image_metadata["target_size"] is None
            ):
                raise Exception(
                    f"Metadata was unavailable for image: {image_metadata['image_path']}. Ensure --skip_file_discovery=metadata is not set."
                )

            if self.print_names:
                logger.info(
                    f"Dataset is now using image: {image_metadata['image_path']}"
                )

            output_data["training_samples"].append(image_metadata)

            if "instance_prompt_text" not in image_metadata:
                raise ValueError(
                    f"Instance prompt text must be provided in image metadata. Image metadata: {image_metadata}"
                )
        return output_data
