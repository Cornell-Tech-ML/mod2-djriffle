from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Type

from typing_extensions import Protocol

from . import operators
from .tensor_data import (
    shape_broadcast,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Shape, Storage, Strides


class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        """Call a map function"""
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Map placeholder"""
        ...

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[[Tensor, Tensor], Tensor]:
        """Zip placeholder"""
        ...

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """Reduce placeholder"""
        ...

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Matrix multiply"""
        raise NotImplementedError("Not implemented in this assignment")

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
        ----
            ops : tensor operations object see `tensor_ops.py`


        Returns:
        -------
            A collection of tensor functions

        """
        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
        ----
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
        -------
            new tensor data

        """
        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float],
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
        -------
            :class:`TensorData` : new tensor data

        """
        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)  # type: ignore
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(  # noqa: D417
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
        ----
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
        -------
            :class:`TensorData` : new tensor

        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        """Matrix multiplication"""
        raise NotImplementedError("Not implemented in this assignment")

    is_cuda = False


# Implementations.


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
    ----
        fn: function from float-to-float to apply

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # Calculate the total number of elements in the output tensor.
        out_size = 1
        for s in out_shape:
            out_size *= s

        # Iterate over every element in the output tensor.
        for idx in range(out_size):
            # Compute the multidimensional index for the output tensor.
            out_index = [0] * len(out_shape)
            cur_idx = idx
            for i in range(len(out_shape) - 1, -1, -1):
                out_index[i] = cur_idx % out_shape[i]
                cur_idx //= out_shape[i]

            # Broadcast input index
            in_index = [0] * len(in_shape)
            for i in range(len(in_shape)):
                if in_shape[i] == 1:
                    in_index[i] = 0
                else:
                    in_index[i] = out_index[i]

            # Compute flat positions in both storage arrays.
            out_pos = sum(out_index[i] * out_strides[i] for i in range(len(out_shape)))
            in_pos = sum(in_index[i] * in_strides[i] for i in range(len(in_shape)))

            # Apply the function and store the result.
            out[out_pos] = fn(in_storage[in_pos])

    return _map


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
    ----
        fn: function mapping two floats to float to apply

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # Calculate the total number of elements in the output tensor.
        out_size = 1
        for s in out_shape:
            out_size *= s

        # Iterate over every element in the output tensor.
        for idx in range(out_size):
            # Compute the multidimensional index for the output tensor.
            out_index = [0] * len(out_shape)
            cur_idx = idx
            for i in range(len(out_shape) - 1, -1, -1):
                out_index[i] = cur_idx % out_shape[i]
                cur_idx //= out_shape[i]

            # Broadcast input indices
            a_index = [0] * len(a_shape)
            for i in range(len(a_shape)):
                if a_shape[i] == 1:
                    a_index[i] = 0
                else:
                    a_index[i] = out_index[i]

            b_index = [0] * len(b_shape)
            for i in range(len(b_shape)):
                if b_shape[i] == 1:
                    b_index[i] = 0
                else:
                    b_index[i] = out_index[i]

            # Compute flat positions in all storage arrays.
            out_pos = sum(out_index[i] * out_strides[i] for i in range(len(out_shape)))
            a_pos = sum(a_index[i] * a_strides[i] for i in range(len(a_shape)))
            b_pos = sum(b_index[i] * b_strides[i] for i in range(len(b_shape)))

            # Apply the function and store the result.
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return _zip


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
    ----
        fn: reduction function mapping two floats to float

    Returns:
    -------
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # TODO: Implement for Task 2.3.
        # Calculate the total number of elements in the output tensor.
        out_size = 1
        for s in out_shape:
            out_size *= s

        # Iterate over every element in the output tensor.
        for idx in range(out_size):
            # Compute the multidimensional index for the output tensor.
            out_index = [0] * len(out_shape)
            cur_idx = idx
            for i in range(len(out_shape) - 1, -1, -1):
                out_index[i] = cur_idx % out_shape[i]
                cur_idx //= out_shape[i]

            # Compute flat position in the output storage.
            out_pos = sum(out_index[i] * out_strides[i] for i in range(len(out_shape)))

            # Initialize the output element to the start value.
            out[out_pos] = a_storage[
                out_pos
            ]  # Starting from the first element in the reduction dim

            # Iterate over the reduction dimension.
            for r in range(1, a_shape[reduce_dim]):
                # Update the index along the reduction dimension.
                a_index = list(out_index)
                a_index[reduce_dim] = r

                # Compute flat position in the input storage.
                a_pos = sum(a_index[i] * a_strides[i] for i in range(len(a_shape)))

                # Apply the reduction function.
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)
