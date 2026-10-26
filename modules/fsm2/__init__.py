"""FSM2 in Python with an in-memory API.

Example
-------
.. code-block:: python

    from fsm2 import fsm2, Forzamiento, CondicionesIniciales, Configuracion
    simulacion, condiciones_finales = fsm2(
        forzamiento, condiciones_iniciales, configuracion
    )
"""

from .api import (
    Condiciones,
    CondicionesFinales,
    CondicionesIniciales,
    Configuracion,
    Forzamiento,
    Simulacion,
    compilar_numba,
    fsm2,
)

__all__ = [
    "fsm2",
    "compilar_numba",
    "Forzamiento",
    "Condiciones",
    "CondicionesIniciales",
    "CondicionesFinales",
    "Configuracion",
    "Simulacion",
]

__version__ = "2.1.2"
