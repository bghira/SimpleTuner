import torch
from torch import Tensor, FloatTensor


def swap_first_and_last_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Swap the first dimension with the last dimension of a tensor.

    Args:
        tensor (torch.Tensor): The input tensor of any shape.

    Returns:
        torch.Tensor: A tensor with the first dimension swapped with the last.
    """
    # Get the total number of dimensions
    num_dims = len(tensor.shape)

    # Create a new order of dimensions
    new_order = list(range(1, num_dims)) + [0]

    # Permute the tensor according to the new order
    return tensor.permute(*new_order)


def swap_back_first_and_last_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Swap back the first dimension with the last dimension of a tensor
    to its original shape after a swap.

    Args:
        tensor (torch.Tensor): The tensor that had its first and last dimensions swapped.

    Returns:
        torch.Tensor: A tensor with its original shape restored.
    """
    # Get the total number of dimensions
    num_dims = len(tensor.shape)

    # Create a new order to reverse the previous swapping
    new_order = [num_dims - 1] + list(range(0, num_dims - 1))

    # Permute the tensor according to the new order
    return tensor.permute(*new_order)


def copy_stochastic_(target: Tensor, source: Tensor):
    """
    copies source into target using stochastic rounding

    Args:
        target: the target tensor with dtype=bfloat16
        source: the target tensor with dtype=float32
    """
    # create a random 16 bit integer
    result = torch.randint_like(
        source,
        dtype=torch.int32,
        low=0,
        high=(1 << 16),
    )

    # add the random number to the lower 16 bit of the mantissa
    result.add_(source.view(dtype=torch.int32))

    # mask off the lower 16 bit of the mantissa
    result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

    # copy the higher 16 bit into the target tensor
    target.copy_(result.view(dtype=torch.float32))

    del result


def add_stochastic_(_input: Tensor, other: Tensor, alpha: float = 1.0):
    """
    Adds other to input using stochastic rounding.

    There is a hack to fix a bug on MPS where uneven final dimensions cause
    a crash.

    Args:
        _input: the input tensor with dtype=bfloat16
        other: the other tensor
        alpha: a multiplier for other
    """
    _input_original = _input
    if _input.device.type == "mps":
        _input = _input.to(dtype=torch.float32)

    if other.dtype == torch.float32:
        result = other.clone()
    else:
        result = other.to(dtype=torch.float32)

    if _input.device.type == "mps":
        result.add_(_input, alpha=torch.tensor(alpha, dtype=torch.float32))
    else:
        result.add_(_input, alpha=alpha)

    copy_stochastic_(_input, result)

    if _input.device.type == "mps":
        _input_original.copy_(_input.view(dtype=torch.float32))


def addcdiv_stochastic_(
    _input: Tensor, tensor1: Tensor, tensor2: Tensor, value: float = 1.0
):
    """
    adds (tensor1 / tensor2 * value) to input using stochastic rounding

    Args:
        _input: the input tensor with dtype=bfloat16
        tensor1: the numerator tensor
        tensor2: the denominator tensor
        value: a multiplier for tensor1/tensor2
    """
    if _input.dtype == torch.float32:
        result = _input.clone()
    else:
        result = _input.to(dtype=torch.float32)

    result.addcdiv_(tensor1, tensor2, value=value)
    copy_stochastic_(_input, result)
