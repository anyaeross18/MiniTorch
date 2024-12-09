"""Collection of the core mathematical operators used throughout the code base."""

# from itertools import accumulate
import math

# ## Task 0.1
# from types import NoneType
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


def mul(x: float, y: float) -> float:
    """Returns the product of x and y"""
    return x * y


def id(i: float) -> float:
    """Returns the id, i, unchanged"""
    return i


def add(x: float, y: float) -> float:
    """Returns the sum of x and y"""
    return x + y


def neg(x: float) -> float:
    """Returns the negation of x"""
    return -x


def lt(lessThan: float, greaterThan: float) -> float:
    """Returns true if number variable lessThan is less than number variable greaterThan"""
    return 1.0 if lessThan < greaterThan else 0.0


def eq(x: float, y: float) -> float:
    """Returns true if x equals y"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the greater value between x and y"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Returns if the difference between 2 numbers is less than 0.01"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Generates the sigmoid of a number
    Args:
    x (float): sigmoid of variable is calculated

    Returns
    -------
    The sigmoid for x, sigmoid(x) = 1/(1+e^-x) when x is greater than or
    equal 0 or sigmoid(x) = (e^x)/(1+e^x) when x is less than 0

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Returns the ReLU function; returns 0 if x is less than or equal to 0 else returns the value of x"""
    return x if x > 0 else 0.0


ESP = 1e-6


def log(x: float) -> float:
    """Return the natural logorithim of x"""
    return math.log(x + ESP)


def exp(x: float) -> float:
    """Returns e, Euler's number, raised to the power of x"""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the reciprocal of x, (1/x), if x is not 0"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Returns the derivative of the natural log of x, (1/x), times the varible y"""
    return y / (x + ESP)


def inv_back(x: float, y: float) -> float:
    """Returns the derivative of the reciprocal of x, (1/(x^2)), times y"""
    return -y / (x**2)


def relu_back(x: float, y: float) -> float:
    """Returns the derivative of the ReLU function times the variable y"""
    return y if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.


def map(func: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Applies a given function to each element in an iterable and returns an iterable of the results
    Args:
    func (Callable[[float], float]): A function that takes a single float as input and returns a float
    data (Iterable[float]): An iterable of float values

    Returns
    -------
    An Iterable of elements with the function, `func`, applied to each element in data

    """

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(func(x))
        return ret

    return _map


def zipWith(
    func: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Combines two iterables element-wise using a given function, `func`, and fills the smaller array with `fillVal`
    Args:
    func (Callable[[float, float], float]): a function with two float inputs and one float output
    data1 (Iterable[float]): An iterable of float values
    data2 (Iterable[float]): An iterable of float values
    fillVal (float): The value used to fill in the shorter iterable to match the length of the longer one. Default is 0.0

    Returns
    -------
    An iterable of float values where each element is the result of applying `func` to corresponding elements from `data1` and `data2`.
    If the iterables are of different lengths, the shorter one is extended with `fillVal` to match the length of the longer one.

    """

    def _zipWith(data1: Iterable[float], data2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(data1, data2):
            ret.append(func(x, y))
        return ret

    return _zipWith


def reduce(
    func: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Reduces an iterable of float values to a single float by applying a given function, `func`, cumulatively
    Args:
    func (Callable[[float, float], float]): A function that takes two float arguments and returns a float
    data1 (Iterable[float]): An iterable of float values to be reduced

    Returns
    -------
    A single float value calculated by applying `func` to each element of `data1`
    If `data1` is empty, 0 is returned
    If `data1` contains only one element, that element is returned without applying `func`

    """

    def _reduce(data: Iterable[float]) -> float:
        val = start
        for x in data:
            val = func(val, x)
        return val

    return _reduce


def negList(data: Iterable[float]) -> Iterable[float]:
    """Returns the negation of all elements in an iterable"""
    return map(neg)(data)


def addLists(data1: Iterable[float], data2: Iterable[float]) -> Iterable[float]:
    """Returns an iterable where corresponding elements from `data1` and `data2` are added together"""
    return zipWith(add)(data1, data2)


def sum(data: Iterable[float]) -> float:
    """Returns a float representing the sum of all elements in an iterable"""
    return reduce(add, 0.0)(data)


def prod(data: Iterable[float]) -> float:
    """Returns a float representing the product of all elements in an iterable"""
    return reduce(mul, 1.0)(data)


# Task 0.3.
