"""Physical constants used by FSM2.

The physical values are the same constants defined in
``src/FSM2_MODULES.F90``.  The Python implementation uses ``float64`` for
the state and calculations, but keeps the original numerical constants so
that results remain directly comparable with the Fortran model.  ``EPSR`` is
the exception: it is the machine epsilon of ``float64`` because the reference
Fortran build also uses double precision.
"""

CP = 1005.0
EPS = 0.622
E0 = 611.213
G = 9.81
HCAP_ICE = 2100.0
HCAP_WAT = 4180.0
HCON_AIR = 0.025
HCON_CLAY = 1.16
HCON_ICE = 2.24
HCON_SAND = 1.57
HCON_WAT = 0.56
I0 = 1367.0
LF = 0.334e6
LV = 2.501e6
LS = LF + LV
MU_WAT = 1.78e-3
PI = 3.14159
RAIR = 287.0
RWAT = 462.0
RHO_ICE = 917.0
RHO_WAT = 1000.0
SB = 5.67e-8
TM = 273.15
VKMAN = 0.4

# Fortran's intrinsic ``epsilon(x)`` is the machine epsilon of the type of
# ``x``.  The reference model is compiled here in double precision, and this
# implementation stores all state in ``float64``, so comparisons written as
# ``epsilon(...)`` in the sources use the float64 value.
EPSR = 2.220446049250313e-16
