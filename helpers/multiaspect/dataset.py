from torch.utils.data import Dataset
from pathlib import Path
import logging, os

logger = logging.getLogger("MultiAspectDataset")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))


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
        print_names=False,
    ):
        self.id = id
        self.datasets = datasets
        self.print_names = print_names

    def __len__(self):
        # Sum the length of all data backends:
        return sum([len(dataset) for dataset in self.datasets])

    def __getitem__(self, image_tuple):
        output_data = []
        for sample in image_tuple:
            image_metadata = sample
            if (
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

            if "instance_prompt_text" not in image_metadata:
                raise ValueError(
                    f"Instance prompt text must be provided in image metadata. Image metadata: {image_metadata}"
                )
            output_data.append(image_metadata)

        return output_data
