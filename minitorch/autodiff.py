from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


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
    vals_plus = list(vals)
    vals_minus = list(vals)

    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    derivative = (f_plus - f_minus) / (2 * epsilon)

    return derivative


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate a derivative for this variable."""
        pass

    @property
    def unique_id(self) -> int:
        """A unique identifier for this variable."""
        ...

    def is_leaf(self) -> bool:
        """True if this variable is a leaf (no parents)."""
        ...

    def is_constant(self) -> bool:
        """True if this variable is a constant (no `last_fn` and no `derivative`)"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Returns the parents of this variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Backpropagate derivatives through the computation graph."""
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
    # TODO: Implement for Task 1.4.
    visited = set()
    order = []

    def dfs(v: Variable) -> None:
        if v.unique_id in visited or v.is_constant():
            return
        visited.add(v.unique_id)
        for parent in v.parents:
            dfs(parent)
        order.append(v)

    dfs(variable)
    return reversed(order)  # Reverse to get the correct topological order


def backpropagate(variable: Variable, deriv: Any) -> None:  # noqa: D417
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    # TODO: Implement for Task 1.4.
    # Get variables in topological order
    topo_order = list(topological_sort(variable))

    # Map from variable unique_id to derivative
    derivatives = {v.unique_id: 0.0 for v in topo_order}
    derivatives[variable.unique_id] = deriv

    # Traverse variables in reverse topological order (from output to inputs)
    for v in topo_order:
        d_output = derivatives[v.unique_id]
        if v.is_leaf():
            v.accumulate_derivative(d_output)
        else:
            # Compute gradients with respect to parents
            for parent, grad in v.chain_rule(d_output):
                if parent.unique_id in derivatives:
                    derivatives[parent.unique_id] += grad
                else:
                    derivatives[parent.unique_id] = grad


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
        """Returns the saved values."""
        return self.saved_values
