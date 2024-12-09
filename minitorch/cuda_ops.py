# type: ignore
# Currently pyright doesn't support numba.cuda

from typing import Callable, Optional, TypeVar, Any

import numba
from numba import cuda
from numba.cuda import jit as _jit
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

FakeCUDAKernel = Any

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

Fn = TypeVar("Fn")


def device_jit(fn: Fn, **kwargs: Any) -> Fn:
    """JIT compile a function for CUDA device.

    Args:
    ----
        fn: The function to compile.
        **kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        The JIT compiled function.

    """
    return _jit(device=True, **kwargs)(fn)  # type: ignore


def jit(fn: Fn, **kwargs: Any) -> FakeCUDAKernel:
    """JIT compile a function for CUDA.

    Args:
    ----
        fn: The function to compile.
        **kwargs: Additional keyword arguments for the JIT compiler.

    Returns:
    -------
        The JIT compiled function.

    """
    return _jit(**kwargs)(fn)  # type: ignore


to_index = device_jit(to_index)
index_to_position = device_jit(index_to_position)
broadcast_index = device_jit(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        cufn: Callable[[float], float] = device_jit(fn)
        f = tensor_map(cufn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """A method to zip two tensors element-wise using a given function."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_zip(cufn)

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """A method to reduce a tensor along a specified dimension using a given function."""
        cufn: Callable[[float, float], float] = device_jit(fn)
        f = tensor_reduce(cufn)

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Perform matrix multiplication on two tensors using CUDA.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the matrix multiplication.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out

    @staticmethod
    def conv1d(a: Tensor, b: Tensor) -> Tensor:
        """Perform convolution on two tensors using CUDA.

        Args:
        ----
            a (Tensor): The first input tensor.
            b (Tensor): The second input tensor.

        Returns:
        -------
            Tensor: The result of the convolution.

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        # assert a.shape[-1] == b.shape[-1]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tens_cuda_conv[blockspergrid, threadsperblock](
            *out.tuple(), *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        in_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            position_in = index_to_position(in_index, in_strides)
            position_out = index_to_position(out_index, out_strides)
            out[position_out] = fn(in_storage[position_in])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
    ----
        fn: function mappings two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        a_index = cuda.local.array(MAX_DIMS, numba.int32)
        b_index = cuda.local.array(MAX_DIMS, numba.int32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

        if i < out_size:
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            position_a = index_to_position(a_index, a_strides)
            position_b = index_to_position(b_index, b_strides)
            position_out = index_to_position(out_index, out_strides)
            out[position_out] = fn(a_storage[position_a], b_storage[position_b])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    r"""Implement a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    cache = cuda.shared.array(BLOCK_DIM, numba.float64)
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    pos = cuda.threadIdx.x

    if pos < BLOCK_DIM:
        cache[pos] = 0.0

    if i < size:
        cache[pos] = a[i]
    cuda.syncthreads()

    if i < size:
        j = 1
        while j < BLOCK_DIM:
            if pos % (2 * j) == 0:
                cache[pos] += cache[pos + j]
                cuda.syncthreads()
            j *= 2

    if pos == 0:
        out[cuda.blockIdx.x] = cache[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    """Practice function to sum elements of a tensor using CUDA.

    Args:
    ----
        a (Tensor): Input tensor to be summed.

    Returns:
    -------
        TensorData: Resulting tensor after summation.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """CUDA higher-order tensor reduce function.

    Args:
    ----
        fn: reduction function maps two floats to float.

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024
        cache = cuda.shared.array(BLOCK_DIM, numba.float64)
        out_index = cuda.local.array(MAX_DIMS, numba.int32)
        out_pos = cuda.blockIdx.x
        pos = cuda.threadIdx.x

        if pos < BLOCK_DIM:
            cache[pos] = reduce_value

        if out_pos < out_size:
            to_index(out_pos, out_shape, out_index)
            position_out = index_to_position(out_index, out_strides)
            out_index[reduce_dim] = out_index[reduce_dim] * BLOCK_DIM + pos
            position_a = index_to_position(out_index, a_strides)

            if out_index[reduce_dim] < a_shape[reduce_dim]:
                cache[pos] = a_storage[position_a]
                cuda.syncthreads()
                j = 1
                while pos % (2 * j) == 0 and j < BLOCK_DIM:
                    cache[pos] = fn(cache[pos], cache[pos + j])
                    cuda.syncthreads()
                    j *= 2

            if pos == 0:
                out[position_out] = cache[0]

    return jit(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """Implement a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
    ----
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square

    """
    # BLOCK_DIM defines the maximum block dimension
    BLOCK_DIM = 32

    # Allocate shared memory for storing sub-matrices of `a` and `b`.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # Compute global thread indices for this CUDA thread.
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Row index for the thread
    j = (
        cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    )  # Column index for the thread

    # Check if the thread's indices are within the matrix bounds.
    # Ensures no out-of-bounds memory when size < BLOCK_DIM.
    if i < size and j < size:
        # Move data from global memory into shared memory for `a` and `b`.
        # This meets the requirement to first load all data into shared memory.
        a_shared[cuda.threadIdx.x, cuda.threadIdx.y] = a[i * size + j]
        b_shared[cuda.threadIdx.x, cuda.threadIdx.y] = b[i * size + j]

        # Synchronize all threads in the block to ensure shared memory is fully populated.
        cuda.syncthreads()

        # Initialize a temporary variable to accumulate the result of the dot product.
        temp = 0

        # Compute the dot product of the corresponding row from `a_shared`
        # and column from `b_shared` for the given `i` and `j`.
        # This step computes one element of the output matrix `out`.
        for k in range(size):
            # Multiply the element from the current row of `a_shared` with
            # the corresponding element from the current column of `b_shared`.
            # Accumulate the results in `temp`.
            temp += a_shared[cuda.threadIdx.x, k] * b_shared[k, cuda.threadIdx.y]

        # Write the result to global memory once per kernel.
        # This meets the 3rd requirement to have one global memory write per kernel.
        out[i * size + j] = temp


jit_mm_practice = jit(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    """Practice function to perform matrix multiplication using CUDA.

    Args:
    ----
        a (Tensor): The first input tensor.
        b (Tensor): The second input tensor.

    Returns:
    -------
        TensorData: The result of the matrix multiplication.

    """
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    # Ensure that the dimensions of `a` and `b` are compatible for matrix multiplication.
    assert a_shape[-1] == b_shape[-2]

    # Compute batch strides for both tensors (if applicable).
    # These are used to handle batched matrix multiplication.
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Batch dimension - fixed
    batch = cuda.blockIdx.z

    # Block dimension (BLOCK_DIM) for shared memory allocation.
    BLOCK_DIM = 32

    # Allocate shared memory for storing sub-matrices of `a` and `b`.
    a_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)
    b_shared = cuda.shared.array((BLOCK_DIM, BLOCK_DIM), numba.float64)

    # The final position c[i, j]
    # Compute global thread indices for this CUDA thread.
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # Row index for the thread
    j = (
        cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    )  # Column index for the thread

    # The local position in the block.
    pi = cuda.threadIdx.x  # Local row index for the thread
    pj = cuda.threadIdx.y  # Local column index for the thread

    # Code Plan:
    # 1) Move across shared dimension by block dim.
    #    a) Copy into shared memory for a matrix.
    #    b) Copy into shared memory for b matrix
    #    c) Compute the dot produce for position c[i, j]

    # Initialize a temporary variable to accumulate the result of the dot product.
    temp = 0.0

    # Iterate over sub-matrices of `a` and `b` to perform the matrix multiplication.
    for k in range(0, a_shape[-1], BLOCK_DIM):
        # Satisfies Requirement 1: Move data from global memory into shared memory for `a` and `b`.
        # Satisfies Requirement 2: Only read each cell in `a` and `b` once.
        # Requirement 2: each kernel read only once per iteration and does not go back to 'a' or 'b' for the same data.

        # Load a sub-matrix of `a_storage` into shared memory `a_shared`.
        # Each thread reads the relevant and unique element of `a` based on its block and thread indices.
        # Conditions ensure no out-of-bounds memory access for rows (`i`) or columns (`k + pj`).
        # Indexing considers batch offset, current row, and column position.
        if i < out_shape[1] and k + pj < a_shape[-1]:
            a_shared[pi, pj] = a_storage[
                (batch * a_batch_stride) + (i * a_strides[1]) + (k + pj) * a_strides[2]
            ]

        # Load a sub-matrix of `b_storage` into shared memory `b_shared`.
        # Ensures threads load the relevant and unique element of `b_storage` into shared memory based on indices.
        # Conditions ensure no out-of-bounds memory access for rows (`k + pi`) or columns (`j`).
        # Indexing considers batch offset, current row, and column position.
        if j < out_shape[2] and k + pi < b_shape[-2]:
            b_shared[pi, pj] = b_storage[
                (batch * b_batch_stride) + (k + pi) * b_strides[1] + j * b_strides[2]
            ]

        cuda.syncthreads()  # Synchronize all threads to ensure data is loaded

        # Perform the dot product for the tile, accessing each value in shared memory exactly once.
        for kk in range(BLOCK_DIM):
            if (
                kk + k < a_shape[-1]
            ):  # Ensure we're within valid bounds for `a` and `b`.
                temp += a_shared[pi, kk] * b_shared[kk, pj]  # Compute the dot product

    # Satisfaction of Requirement 3: Only write to global memory once per kernel.
    # Ensure that the thread's indices are within the matrix bounds.
    # Write the result of the dot product to the output tensor `out`.
    if i < out_shape[1] and j < out_shape[2]:
        out[(batch * out_strides[0]) + (i * out_strides[1]) + (j * out_strides[2])] = (
            temp
        )


tensor_matrix_multiply = jit(_tensor_matrix_multiply)


def cuda_conv1D(
    in_storage: Storage,
    w_storage: Storage,
    out_storage: Storage,
    in_shape: Shape,
    w_shape: Shape,
    out_shape: Shape,
    in_strides: Strides,
    w_strides: Strides,
    out_strides: Strides,
) -> None:
    """Perform convolution on CUDA device.

    Args:
    ----
        in_storage (Storage): Input tensor storage.
        w_storage (Storage): Weight tensor storage.
        out_storage (Storage): Output tensor storage.
        in_shape (Shape): Shape of the input tensor.
        w_shape (Shape): Shape of the weight tensor.
        out_shape (Shape): Shape of the output tensor.
        in_strides (Strides): Strides of the input tensor.
        w_strides (Strides): Strides of the weight tensor.
        out_strides (Strides): Strides of the output tensor.

    Returns:
    -------
        None

    """
    # Define BLOCK_SIZE and other necessary constants
    BLOCK_SIZE = THREADS_PER_BLOCK

    batch_, out_channels, out_width = out_shape
    batch, in_channels, width = in_shape
    out_channels_, in_channels_, kw = w_shape

    assert (
        batch == batch_
        and in_channels == in_channels_
        and out_channels == out_channels_
    )

    s1 = in_strides
    s2 = w_strides

    # Define shared memory
    in_shared = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=numba.float32)
    w_shared = cuda.shared.array((BLOCK_SIZE, BLOCK_SIZE), dtype=numba.float32)

    # Calculate thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    # Calculate global indices
    x = tx + bx * cuda.blockDim.x
    y = ty + by * cuda.blockDim.y

    if x >= out_width or y >= batch:
        return

    # Initialize the output value
    out_value = 0.0

    # put input and weight into shared memory
    for i in range(0, in_channels, BLOCK_SIZE):
        in_shared[ty, tx] = in_storage[x * s1[2] + (i + tx) * s1[1] + y * s1[0] + ty]
        w_shared[ty, tx] = w_storage[(i + ty) * s2[2] + tx * s2[1] + y * s2[0]]

        cuda.syncthreads()

        for i in range(0, BLOCK_SIZE):
            out_value += in_shared[ty, i] * w_shared[i, tx]

        cuda.syncthreads()

    out_storage[x * out_strides[2] + y * out_strides[0] + tx] = out_value


tens_cuda_conv = jit(cuda_conv1D)


def conv1_(a: Tensor) -> TensorData:
    """Practice function to sum elements of a tensor using CUDA.

    Args:
    ----
        a (Tensor): Input tensor to be summed.

    Returns:
    -------
        TensorData: Resulting tensor after summation.

    """
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    tens_cuda_conv[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out
