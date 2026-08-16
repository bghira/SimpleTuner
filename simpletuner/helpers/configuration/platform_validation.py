import torch


def validate_mps_train_batch_size(train_batch_size: int) -> None:
    """Reject training batch sizes that exceed the existing MPS safety limit."""
    if torch.backends.mps.is_available() and train_batch_size > 16:
        raise ValueError(
            "An M3 Max 128G will use 12 seconds per step at a batch size of 1 and 65 seconds per step at a batch size of 12."
            " Any higher values will result in NDArray size errors or other unstable training results and crashes."
            "\nPlease reduce the batch size to 12 or lower."
        )
