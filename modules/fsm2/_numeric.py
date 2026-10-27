"""Small numerical helpers shared by the physics routines."""

from numba import njit


@njit(cache=True, inline="always", fastmath=True)
def array_sum(values):
    """Sum an array in index order, as Fortran's intrinsic ``sum`` does."""
    total = 0.0
    for i in range(values.shape[0]):
        total += values[i]
    return total


# gfortran expands integer powers into multiplications: x**3 as (x*x)*x and
# x**4 as (x*x)*(x*x).  Python's ``**`` calls ``pow`` instead, which can differ
# in the last bit, so use the same expansion explicitly.
@njit(cache=True, inline="always", fastmath=True)
def cube(x):
    return (x * x) * x


@njit(cache=True, inline="always", fastmath=True)
def fourth(x):
    return (x * x) * (x * x)


@njit(cache=True, inline="always", fastmath=True)
def sixth(x):
    return cube(x) * cube(x)
