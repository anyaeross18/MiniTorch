from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the scalar function to the given values.

        Args:
        ----
            vals (ScalarLike): The input values to the function.

        Returns:
        -------
            Scalar: The result of applying the function.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


class Mul(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Returns the product of x and y"""
        ctx.save_for_backward(x, y)
        return operators.mul(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns the derivatives of the product of x and y"""
        x, y = ctx.saved_values
        return d_output * y, d_output * x


class Inv(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Returns the inverse of x"""
        ctx.save_for_backward(x)
        return operators.inv(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the derivative of the inverse of x"""
        (a,) = ctx.saved_values
        return -d_output / (a * a)


class Neg(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Returns the negation of x"""
        return operators.neg(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the derivative of the negation of x"""
        return -d_output


class Sigmoid(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Returns the sigmoid of x"""
        result = operators.sigmoid(x)
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the derivative of the sigmoid of x"""
        (sigmoid_x,) = ctx.saved_values
        return d_output * sigmoid_x * (1 - sigmoid_x)


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Returns the ReLU of x"""
        ctx.save_for_backward(x)
        return operators.relu(x)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the derivative of the ReLU of x"""
        (a,) = ctx.saved_values
        return d_output * (1.0 if a > 0 else 0.0)


class Exp(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float) -> float:
        """Returns the exponential of x"""
        out = operators.exp(x)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Returns the derivative of the exponential of x"""
        (a,) = ctx.saved_values
        return a * d_output


class LT(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Returns true if x is less than y"""
        return operators.lt(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns the derivatives of the less than function"""
        return 0.0, 0.0


class EQ(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, x: float, y: float) -> float:
        """Returns true if x equals y"""
        return operators.eq(x, y)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Returns the derivatives of the equality function"""
        return 0.0, 0.0


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Add two numbers"""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Derivatives of addition"""
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Log of a"""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Derivative of log"""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.
