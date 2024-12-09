from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative value `x` for this variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Returns the unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is a constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parent variables of the current variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """Returns the gradient for the parents of the variable."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()  # To keep track of visited nodes
    stack: List[Variable] = []  # Stack to hold the topological order

    def visit(node: Variable) -> None:
        if node.is_constant() or node.unique_id in visited:
            return
        if not node.is_leaf():
            for m in node.parents:
                if not m.is_constant():
                    visit(m)
        visited.add(node.unique_id)
        stack.insert(0, node)

    visit(variable)  # Start the visit from the given variable
    return stack


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable.
    deriv: The initial derivative value of the right-most variable that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # Get the topological order of the computation graph
    ordered_vars = topological_sort(variable)

    # Dictionary to accumulate derivatives for each variable
    derivatives = {}
    derivatives[variable.unique_id] = deriv  # Set the initial derivative

    for var in ordered_vars:
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            # Calculate the derivatives for the parents using the chain rule
            dout = derivatives[var.unique_id]
            for parent, parent_derivative in var.chain_rule(dout):
                if parent.is_constant():
                    continue
                if parent.unique_id not in derivatives:
                    derivatives[parent.unique_id] = 0.0  # Initialize if not present
                derivatives[parent.unique_id] += parent_derivative


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved tensors for backpropagation."""
        return self.saved_values
