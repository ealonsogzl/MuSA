"""Canopy snow interception and unloading (translation of ``INTERCEPT.F90``)."""

import math

from numba import njit

from ._constants import EPSR, LF, TM


@njit(cache=False, fastmath=True)
def intercept(params, dt, cveg, Eveg, lveg, Scap, Sf, Ta, Ua, Sveg, Tveg):
    """Mass balance of snow intercepted by vegetation."""
    ncnpy = params.ncnpy

    drip = 0.0
    Sfg = Sf
    svg = 0.0
    unload = 0.0

    if lveg[0] > EPSR:
        for n in range(ncnpy):
            # Interception of falling snow
            fveg = 1.0 - math.exp(-params.kext * lveg[n])
            if params.canint == 1:
                dsvg = fveg * Sfg * dt
            else:
                dsvg = (Scap[n] - Sveg[n]) * (
                    1.0 - math.exp(-fveg * Sfg * dt / Scap[n])
                )
            if Sveg[n] + dsvg > Scap[n]:
                dsvg = Scap[n] - Sveg[n]
            Sveg[n] = Sveg[n] + dsvg
            Sfg = Sfg - dsvg / dt

            # Sublimation of canopy snow
            if Eveg[n] > 0.0:
                if Sveg[n] > 0.0:
                    Sveg[n] = Sveg[n] - Eveg[n] * dt
                if Sveg[n] < 0.0:
                    Sveg[n] = 0.0
            else:
                if Tveg[n] < TM:  # frost on canopy
                    Sveg[n] = Sveg[n] - Eveg[n] * dt
                    if Sveg[n] > Scap[n]:
                        unload = unload + Sveg[n] - Scap[n]
                        Sveg[n] = Scap[n]

            # Melting of canopy snow
            melt = 0.0
            if Tveg[n] > TM:
                melt = cveg[n] * (Tveg[n] - TM) / LF
                if melt > Sveg[n]:
                    melt = Sveg[n]
                drip = drip + melt
                Sveg[n] = Sveg[n] - melt
                Tveg[n] = Tveg[n] - LF * melt / cveg[n]

            # Unloading of canopy snow
            if params.canunl == 1:
                dsvg = Sveg[n] * dt / params.eunl + params.munl * melt
            else:
                dsvg = (
                    max(Tveg[n] - 270.15, 0.0) / params.Tunl
                    + Ua / params.Uunl
                ) * dt * Sveg[n]
            if dsvg > Sveg[n]:
                dsvg = Sveg[n]
            Sveg[n] = Sveg[n] - dsvg
            unload = unload + dsvg

            Sveg[n] = min(max(Sveg[n], 0.0), Scap[n])
            svg = svg + Sveg[n]

    return drip, Sfg, svg, unload
