"""Solar position helper used when shortwave radiation is partitioned."""

import math

from ._constants import PI


def solar_elevation(year, month, day, hour, lat, noon):
    """Solar elevation (radians), following ``SOLARPOS.F90``.

    ``lat`` is in radians and ``hour``/``noon`` are in hours.
    """
    doy = (
        (7 * int(year)) // 4
        - 7 * (int(year) + (int(month) + 9) // 12) // 4
        + (275 * int(month)) // 9
        + int(day)
        - 30
    )
    dangle = 2.0 * PI * (doy - 1) / 365.0
    declin = (
        0.006918
        - 0.399912 * math.cos(dangle)
        + 0.070257 * math.sin(dangle)
        - 0.006758 * math.cos(2.0 * dangle)
        + 0.000907 * math.sin(2.0 * dangle)
        - 0.002697 * math.cos(3.0 * dangle)
        + 0.001480 * math.sin(3.0 * dangle)
    )
    eqtime = (
        0.000075
        + 0.001868 * math.cos(dangle)
        - 0.032077 * math.sin(dangle)
        - 0.014615 * math.cos(2.0 * dangle)
        - 0.04089 * math.sin(2.0 * dangle)
    ) * (12.0 / PI)
    hangle = (PI / 12.0) * (noon - hour - eqtime)
    sin_elev = (
        math.sin(declin) * math.sin(lat)
        + math.cos(declin) * math.cos(lat) * math.cos(hangle)
    )
    if sin_elev > 1.0:
        sin_elev = 1.0
    elif sin_elev < -1.0:
        sin_elev = -1.0
    return math.asin(sin_elev)
