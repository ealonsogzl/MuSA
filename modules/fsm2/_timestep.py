"""One FSM2 timestep at one point (translation of ``FSM2_TIMESTEP.F90``)."""

from numba import njit

from ._intercept import intercept
from ._radiation import canopy, swrad, thermal
from ._snow import snow, soil
from ._surface import srfb


@njit(cache=True, fastmath=True)
def fsm2_timestep(params, dt, elev, zT, zU, LW, Ps, Qa, Rf, Sdif, Sdir,
                  Sf, Ta, trans, Ua, alb0, vegh, VAI,
                  albs, Tsrf, Dsnw, Nsnow, Qcan, Rgrn, Sice, Sliq, Sveg,
                  Tcan, Tsnow, Tsoil, Tveg, Vsmc,
                  ws):
    """Run all FSM2 physics subroutines for one timestep and one point.

    ``ws`` holds the small work arrays reused across timesteps instead of
    allocating them on every call.  They are zeroed here to keep the same
    semantics as the previous ``np.zeros`` initialisation.
    """
    ncnpy = params.ncnpy
    nsmax = params.nsmax
    nsoil = params.nsoil

    (cveg, fcans, lveg, Scap, tdif, Tveg0,
     ksnow, csoil, ksoil, Eveg, SWveg, Wflx) = ws

    # canopy() fully initialises cveg, fcans, lveg, Scap and Tveg0, and
    # thermal() fully initialises ksnow, csoil and ksoil.  Only the arrays
    # that may be left untouched on some branches need resetting.
    tdif[:] = 0.0
    Eveg[:] = 0.0
    SWveg[:] = 0.0
    Wflx[:] = 0.0

    canopy(params, Sveg, Tveg, VAI, cveg, fcans, lveg, Scap, Tveg0)

    albs, fsnow, SWout, SWsrf, SWsub = swrad(
        params, alb0, Dsnw, dt, elev, fcans, lveg, Sdif, Sdir, Sf, Tsrf,
        albs, SWveg, tdif
    )

    Ds1, gs1, ks1, Ts1 = thermal(
        params, Dsnw, Nsnow, Sice, Sliq, Tsnow, Tsoil, Vsmc, csoil,
        ksnow, ksoil
    )

    (Tsrf, Esrf, Eveg, Gsrf, H, LE, LWout, LWsub, Melt, subl, Tsub,
     Usub) = srfb(
        params, cveg, Ds1, dt, fcans, fsnow, gs1, ks1, lveg, LW, Ps, Qa,
        SWsrf, Sveg, SWveg, Ta, tdif, Ts1, Tveg0, Ua, VAI, vegh,
        zT, zU, Tsrf, Qcan, Sice, Tcan, Tveg, Eveg
    )

    drip, Sfg, svg, unload = intercept(
        params, dt, cveg, Eveg, lveg, Scap, Sf, Ta, Ua, Sveg, Tveg
    )

    Nsnow, Gsoil, Roff, snd, snw = snow(
        params, dt, drip, Esrf, Gsrf, ksnow, ksoil, Melt, Rf, Sfg, Ta,
        trans, Tsrf, unload, Nsnow, Dsnw, Rgrn, Sice, Sliq, Tsnow,
        Tsoil, Wflx
    )

    soil(params, csoil, dt, Gsoil, ksoil, Tsoil)

    return (Nsnow, Tsrf, albs, fsnow, H, LE, LWout, LWsub, Melt, Roff,
            snd, snw, subl, svg, SWout, SWsub, Tsub, Usub, Wflx)
