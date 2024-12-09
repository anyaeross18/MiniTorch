from typing import Tuple


from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor
from minitorch.operators import is_close


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    input = input.contiguous()
    input = input.view(batch, channel, new_height, kh, new_width, kw)
    input = input.permute(0, 1, 2, 4, 3, 5)
    input = input.contiguous()
    input = input.view(batch, channel, new_height, new_width, kh * kw)
    return (input, new_height, new_width)


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply average pooling to input with kernel size kernel.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    input_t, new_height, new_width = tile(input, kernel)
    return input_t.mean(dim=4).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


max_reduce = FastOps.reduce(operators.max, -1e9)
zip_equal = FastOps.zip(operators.eq)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        input: tensor to compute argmax over
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size input.shape with 1s at the argmax and 0s elsewhere.

    """
    # shape = input.shape
    # mask = input.zeros(shape)
    max_num = max_reduce(input, dim)
    mask = input == max_num
    """if mask.sum(dim).all().item() != 1:
        first  = argmax(mask, dim)
        temp = tensor([i]*shape[dim] for i in range(shape[dim]))
        mask = zip_equal(first, temp)"""
    return mask


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward of max function"""
        out = max_reduce(input, int(dim.item()))
        ctx.save_for_backward(input, dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward of max function"""
        input, dim = ctx.saved_tensors
        arg = argmax(input, int(dim.item()))
        mask = arg
        if not is_close(arg.sum(dim).all().item(), 1):
            mask = mask / arg.sum(dim)
        return (grad_output * mask, 0.0)


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction to input with dimension dim.

    Args:
    ----
        input: tensor to compute max over
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size input.shape with the max value at the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax as a tensor.

    Args:
    ----
        input: tensor to compute softmax over
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size input.shape with the softmax value at the specified dimension.

    """
    input = input - max(input, dim)
    input = input.exp()
    return input / input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax as a tensor.

    Args:
    ----
        input: tensor to compute log softmax over
        dim: dimension to reduce

    Returns:
    -------
        Tensor of size input.shape with the log softmax value at the specified dimension.

    """
    input = input - max(input, dim)
    log_sum_exp = input.exp().sum(dim).log()
    return input - log_sum_exp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply max pooling to input with kernel size kernel.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    input_t, new_height, new_width = tile(input, kernel)
    return max_reduce(input_t, 4).view(
        input.shape[0], input.shape[1], new_height, new_width
    )


less_than = FastOps.zip(operators.lt)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise, include an argument to turn off.

    Args:
    ----
        input: input tensor
        p: probability of dropping
        ignore: if True, return input without applying dropout, if False, apply dropout

    Returns:
    -------
        Tensor of size input.shape with some values dropped.

    """
    if ignore:
        return input
    rand_tensor = rand(input.shape)
    keep_prob = tensor([1 - p] * input.size).view(*input.shape)
    mask = less_than(rand_tensor, keep_prob)
    return input * mask
