"""Snow thermodynamics/hydrology and soil temperature (SNOW.F90, SOIL.F90)."""

import math

import numpy as np
from numba import njit

from ._constants import (
    E0,
    EPSR,
    G,
    HCAP_ICE,
    HCAP_WAT,
    LF,
    LS,
    MU_WAT,
    RHO_ICE,
    RHO_WAT,
    RWAT,
    TM,
)
from ._numeric import array_sum, cube, sixth
from ._radiation import tridiag


@njit(cache=True, fastmath=True)
def snow(params, dt, drip, Esrf, Gsrf, ksnow, ksoil, Melt, Rf, Sf, Ta,
         trans, Tsrf, unload, Nsnow, Dsnw, Rgrn, Sice, Sliq, Tsnow,
         Tsoil, Wflx):
    """Snow thermodynamics and hydrology (translation of ``SNOW.F90``)."""
    nsmax = params.nsmax
    nsoil = params.nsoil

    csnow = np.zeros(nsmax)
    U = np.zeros(nsmax)
    a = np.zeros(nsmax)
    b = np.zeros(nsmax)
    c = np.zeros(nsmax)
    rhs = np.zeros(nsmax)
    dTs = np.zeros(nsmax)
    D = np.zeros(nsmax)
    E = np.zeros(nsmax)
    phi = np.zeros(nsmax)
    R = np.zeros(nsmax)
    S = np.zeros(nsmax)
    W = np.zeros(nsmax)
    ksat = np.zeros(nsmax)
    thetar = np.zeros(nsmax)
    thetaw = np.zeros(nsmax)
    theta0 = np.zeros(nsmax)
    dtheta = np.zeros(nsmax)
    Qw = np.zeros(nsmax + 1)

    # No snow
    Gsoil = Gsrf
    Roff = Rf + drip / dt
    for n in range(nsmax):
        Wflx[n] = 0.0

    # Existing snowpack
    if Nsnow > 0:
        # Heat conduction
        for n in range(Nsnow):
            csnow[n] = Sice[n] * HCAP_ICE + Sliq[n] * HCAP_WAT
        if Nsnow == 1:
            U[0] = 2.0 / (
                Dsnw[0] / ksnow[0] + params.dzsoil[0] / ksoil[0]
            )
            dTs[0] = (
                (Gsrf + U[0] * (Tsoil[0] - Tsnow[0])) * dt
                / (csnow[0] + U[0] * dt)
            )
        else:
            U[0] = 2.0 / (
                Dsnw[0] / ksnow[0] + Dsnw[1] / ksnow[1]
            )
            a[0] = 0.0
            b[0] = csnow[0] + U[0] * dt
            c[0] = -U[0] * dt
            rhs[0] = (Gsrf - U[0] * (Tsnow[0] - Tsnow[1])) * dt
            for n in range(1, Nsnow - 1):
                U[n] = 2.0 / (
                    Dsnw[n] / ksnow[n] + Dsnw[n + 1] / ksnow[n + 1]
                )
                a[n] = c[n - 1]
                b[n] = csnow[n] + (U[n - 1] + U[n]) * dt
                c[n] = -U[n] * dt
                rhs[n] = (
                    U[n - 1] * (Tsnow[n - 1] - Tsnow[n]) * dt
                    + U[n] * (Tsnow[n + 1] - Tsnow[n]) * dt
                )
            n = Nsnow - 1
            U[n] = 2.0 / (
                Dsnw[n] / ksnow[n] + params.dzsoil[0] / ksoil[0]
            )
            a[n] = c[n - 1]
            b[n] = csnow[n] + (U[n - 1] + U[n]) * dt
            c[n] = 0.0
            rhs[n] = (
                U[n - 1] * (Tsnow[n - 1] - Tsnow[n]) * dt
                + U[n] * (Tsoil[0] - Tsnow[n]) * dt
            )
            dTs = tridiag(Nsnow, a, b, c, rhs)
        for n in range(Nsnow):
            Tsnow[n] = Tsnow[n] + dTs[n]
        n = Nsnow - 1
        Gsoil = U[n] * (Tsnow[n] - Tsoil[0])

        # Convert melting ice to liquid water
        dSice = Melt * dt
        for n in range(Nsnow):
            coldcont = csnow[n] * (TM - Tsnow[n])
            if coldcont < 0.0:
                dSice = dSice - coldcont / LF
                Tsnow[n] = TM
            if dSice > 0.0:
                if dSice > Sice[n]:
                    dSice = dSice - Sice[n]
                    Dsnw[n] = 0.0
                    Sliq[n] = Sliq[n] + Sice[n]
                    Sice[n] = 0.0
                else:
                    Dsnw[n] = (1.0 - dSice / Sice[n]) * Dsnw[n]
                    Sice[n] = Sice[n] - dSice
                    Sliq[n] = Sliq[n] + dSice
                    dSice = 0.0

        # Remove snow by sublimation
        dSice = Esrf * dt
        if dSice > 0.0:
            for n in range(Nsnow):
                if dSice > Sice[n]:
                    dSice = dSice - Sice[n]
                    Dsnw[n] = 0.0
                    Sice[n] = 0.0
                else:
                    Dsnw[n] = (1.0 - dSice / Sice[n]) * Dsnw[n]
                    Sice[n] = Sice[n] - dSice
                    dSice = 0.0

        # Remove wind-transported snow
        dSice = trans * dt
        if dSice > 0.0:
            for n in range(Nsnow):
                if dSice > Sice[n]:
                    dSice = dSice - Sice[n]
                    Dsnw[n] = 0.0
                    Sice[n] = 0.0
                else:
                    Dsnw[n] = (1.0 - dSice / Sice[n]) * Dsnw[n]
                    Sice[n] = Sice[n] - dSice
                    dSice = 0.0

        # Snow density
        if params.densty == 0:
            for n in range(Nsnow):
                if Dsnw[n] > EPSR:
                    Dsnw[n] = (Sice[n] + Sliq[n]) / params.rfix
        elif params.densty == 1:
            for n in range(Nsnow):
                if Dsnw[n] > EPSR:
                    rhos = (Sice[n] + Sliq[n]) / Dsnw[n]
                    if Tsnow[n] >= TM:
                        if rhos < params.rmlt:
                            rhos = params.rmlt + (
                                rhos - params.rmlt
                            ) * math.exp(-dt / params.trho)
                    else:
                        if rhos < params.rcld:
                            rhos = params.rcld + (
                                rhos - params.rcld
                            ) * math.exp(-dt / params.trho)
                    Dsnw[n] = (Sice[n] + Sliq[n]) / rhos
        else:
            mass = 0.0
            for n in range(Nsnow):
                mass = mass + 0.5 * (Sice[n] + Sliq[n])
                if Dsnw[n] > EPSR:
                    rhos = (Sice[n] + Sliq[n]) / Dsnw[n]
                    rhos = rhos + (
                        rhos * G * mass * dt
                        / (
                            params.eta0
                            * math.exp(-(Tsnow[n] - TM) / 12.4 + rhos / 55.6)
                        )
                        + dt * rhos * params.snda
                        * math.exp(
                            (Tsnow[n] - TM) / 23.8
                            - max(rhos - 150.0, 0.0) / 21.7
                        )
                    )
                    Dsnw[n] = (Sice[n] + Sliq[n]) / rhos
                mass = mass + 0.5 * (Sice[n] + Sliq[n])

        # Snow grain growth
        if params.sgrain == 0:
            for n in range(nsmax):
                Rgrn[n] = params.rgr0
        elif params.sgrain == 1:
            for n in range(Nsnow):
                ggr = 2e-13
                if Tsnow[n] < TM:
                    if Rgrn[n] < 1.50e-4:
                        ggr = 2e-14
                    else:
                        ggr = 7.3e-8 * math.exp(-4600.0 / Tsnow[n])
                Rgrn[n] = Rgrn[n] + dt * ggr / Rgrn[n]
        else:
            for n in range(Nsnow):
                # Empty layers can retain Nsnow > 0 after melt; Fortran would
                # divide by zero here, so only grow grains in real layers.
                if Dsnw[n] <= 0.0:
                    continue
                Tu = Tsrf
                if n > 0:
                    Tu = (
                        Dsnw[n - 1] * Tsnow[n] + Dsnw[n] * Tsnow[n - 1]
                    ) / (Dsnw[n] + Dsnw[n - 1])
                Tl = (
                    params.dzsoil[0] * Tsnow[n] + Dsnw[n] * Tsoil[0]
                ) / (Dsnw[n] + params.dzsoil[0])
                if n < Nsnow - 1:
                    Tl = (
                        Dsnw[n + 1] * Tsnow[n] + Dsnw[n] * Tsnow[n + 1]
                    ) / (Dsnw[n] + Dsnw[n + 1])
                dTdz = abs(Tu - Tl) / Dsnw[n]
                thetaw[n] = Sliq[n] / (RHO_WAT * Dsnw[n])
                if thetaw[n] < 1e-4:
                    dpdT = (
                        (E0 / (RWAT * Tsnow[n] ** 2))
                        * (LS / (RWAT * Tsnow[n]) - 1.0)
                        * math.exp((LS / RWAT) * (1.0 / TM - 1.0 / Tsnow[n]))
                    )
                    qv = 9.2e-5 * sixth(Tsnow[n] / TM) * dpdT * dTdz
                    ggr = 1.25e-7 * min(qv, 1e-6)
                else:
                    ggr = 1e-12 * min(thetaw[n] + 0.05, 0.14)
                Rgrn[n] = Rgrn[n] + dt * ggr / Rgrn[n]

    # Add snowfall and frost to layer 1 with fresh snow density and grain size
    Esnow = 0.0
    if Esrf < 0.0 and Tsrf < TM:
        Esnow = Esrf
    dSice = (Sf - Esnow) * dt
    Dsnw[0] = Dsnw[0] + dSice / params.rhof
    if Sice[0] + dSice > EPSR:
        Rgrn[0] = (Sice[0] * Rgrn[0] + dSice * params.rgr0) / (
            Sice[0] + dSice
        )
    Sice[0] = Sice[0] + dSice

    # Add canopy unloading to layer 1 with bulk snow density and grain size
    rhos = params.rhof
    snw = array_sum(Sice) + array_sum(Sliq)
    snd = array_sum(Dsnw)
    if snd > EPSR:
        rhos = snw / snd
    Dsnw[0] = Dsnw[0] + unload / rhos
    if Sice[0] + unload > EPSR:
        Rgrn[0] = (Sice[0] * Rgrn[0] + unload * params.rgr0) / (
            Sice[0] + unload
        )
    Sice[0] = Sice[0] + unload

    # Add wind-blown snow to layer 1 with wind-packed density and fresh grain
    dSice = -trans * dt
    if dSice > 0.0:
        Dsnw[0] = Dsnw[0] + dSice / params.rhow
        Rgrn[0] = (Sice[0] * Rgrn[0] + dSice * params.rgr0) / (
            Sice[0] + dSice
        )
        Sice[0] = Sice[0] + dSice

    # New snowpack
    if Nsnow == 0 and Sice[0] > 0.0:
        Nsnow = 1
        Rgrn[0] = params.rgr0
        Tsnow[0] = min(Ta, TM)

    # Store state of old layers
    for n in range(nsmax):
        D[n] = Dsnw[n]
        R[n] = Rgrn[n]
        S[n] = Sice[n]
        W[n] = Sliq[n]
    for n in range(Nsnow):
        csnow[n] = Sice[n] * HCAP_ICE + Sliq[n] * HCAP_WAT
        E[n] = csnow[n] * (Tsnow[n] - TM)
    Nold = Nsnow
    snd = array_sum(Dsnw)

    # Initialise new layers
    for n in range(nsmax):
        Dsnw[n] = 0.0
        Rgrn[n] = 0.0
        Sice[n] = 0.0
        Sliq[n] = 0.0
        Tsnow[n] = TM
        U[n] = 0.0
    Nsnow = 0

    if snd > 0.0:
        # Re-assign and count snow layers
        dnew = snd
        Dsnw[0] = dnew
        n = 1
        if Dsnw[0] > params.dzsnow[0]:
            for n in range(1, nsmax + 1):
                Dsnw[n - 1] = params.dzsnow[n - 1]
                dnew = dnew - params.dzsnow[n - 1]
                if dnew <= params.dzsnow[n - 1] or n == nsmax:
                    Dsnw[n - 1] = Dsnw[n - 1] + dnew
                    break
        Nsnow = n

        # Fill new layers from the top downwards
        new = 0
        dnew = Dsnw[0]
        done = False
        for n_old in range(Nold):
            while True:
                if D[n_old] < dnew:
                    # All snow from old layer partially fills new layer
                    Rgrn[new] = Rgrn[new] + S[n_old] * R[n_old]
                    Sice[new] = Sice[new] + S[n_old]
                    Sliq[new] = Sliq[new] + W[n_old]
                    U[new] = U[new] + E[n_old]
                    dnew = dnew - D[n_old]
                    break
                else:
                    # Some snow from old layer fills new layer
                    wt = dnew / D[n_old]
                    Rgrn[new] = Rgrn[new] + wt * S[n_old] * R[n_old]
                    Sice[new] = Sice[new] + wt * S[n_old]
                    Sliq[new] = Sliq[new] + wt * W[n_old]
                    U[new] = U[new] + wt * E[n_old]
                    D[n_old] = (1.0 - wt) * D[n_old]
                    E[n_old] = (1.0 - wt) * E[n_old]
                    S[n_old] = (1.0 - wt) * S[n_old]
                    W[n_old] = (1.0 - wt) * W[n_old]
                    new = new + 1
                    if new >= Nsnow:
                        done = True
                        break
                    dnew = Dsnw[new]
            if done:
                break

        # Diagnose snow layer temperatures
        for n in range(Nsnow):
            csnow[n] = Sice[n] * HCAP_ICE + Sliq[n] * HCAP_WAT
            Tsnow[n] = TM + U[n] / csnow[n]
            Rgrn[n] = Rgrn[n] / Sice[n]

        if params.hydrol == 0:
            # Free-draining snow, no retention or freezing
            Wflx[0] = Roff
            for n in range(Nsnow):
                Roff = Roff + Sliq[n] / dt
                Sliq[n] = 0.0
                if n < Nsnow - 1:
                    Wflx[n + 1] = Roff
        elif params.hydrol == 1:
            # Bucket storage
            if np.max(Sliq) > 0.0 or Rf > 0.0:
                for n in range(Nsnow):
                    phi[n] = 1.0 - Sice[n] / (RHO_ICE * Dsnw[n])
                    if phi[n] < 0.0:
                        phi[n] = 0.0
                    SliqMax = RHO_WAT * Dsnw[n] * phi[n] * params.Wirr
                    Sliq[n] = Sliq[n] + Roff * dt
                    Wflx[n] = Roff
                    Roff = 0.0
                    if Sliq[n] > SliqMax:
                        Roff = (Sliq[n] - SliqMax) / dt
                        Sliq[n] = SliqMax
                    csnow[n] = Sice[n] * HCAP_ICE + Sliq[n] * HCAP_WAT
                    coldcont = csnow[n] * (TM - Tsnow[n])
                    if coldcont > 0.0:
                        dSice = min(Sliq[n], coldcont / LF)
                        Sliq[n] = Sliq[n] - dSice
                        Sice[n] = Sice[n] + dSice
                        Tsnow[n] = Tsnow[n] + LF * dSice / csnow[n]
        else:
            # Gravitational drainage
            if np.max(Sliq) > 0.0 or Rf > 0.0:
                for n in range(nsmax + 1):
                    Qw[n] = 0.0
                Qw[0] = Rf / RHO_WAT
                Roff = 0.0
                for n in range(nsmax):
                    thetaw[n] = 0.0
                for n in range(Nsnow):
                    ksat[n] = (
                        0.31 * (RHO_WAT * G / MU_WAT) * Rgrn[n] ** 2
                        * math.exp(-7.8 * Sice[n] / (RHO_WAT * Dsnw[n]))
                    )
                    phi[n] = 1.0 - Sice[n] / (RHO_ICE * Dsnw[n])
                    thetar[n] = params.Wirr * phi[n]
                    thetaw[n] = Sliq[n] / (RHO_WAT * Dsnw[n])
                    if thetaw[n] > phi[n]:
                        Roff = Roff + RHO_WAT * Dsnw[n] * (
                            thetaw[n] - phi[n]
                        ) / dt
                        thetaw[n] = phi[n]
                dth = dt / params.nhyd
                for i in range(params.nhyd):
                    for n in range(nsmax):
                        theta0[n] = thetaw[n]
                    for j in range(10):
                        for n in range(nsmax):
                            a[n] = 0.0
                            b[n] = 1.0 / dth
                        if thetaw[0] > thetar[0]:
                            b[0] = (
                                1.0 / dth
                                + 3.0 * ksat[0]
                                * (thetaw[0] - thetar[0]) ** 2
                                / cube(phi[0] - thetar[0]) / Dsnw[0]
                            )
                            Qw[1] = ksat[0] * cube(
                                (thetaw[0] - thetar[0])
                                / (phi[0] - thetar[0])
                            )
                        rhs[0] = (
                            (thetaw[0] - theta0[0]) / dth
                            + (Qw[1] - Qw[0]) / Dsnw[0]
                        )
                        for n in range(1, Nsnow):
                            if thetaw[n - 1] > thetar[n - 1]:
                                a[n] = (
                                    -3.0 * ksat[n - 1]
                                    * (thetaw[n - 1] - thetar[n - 1]) ** 2
                                    / cube(phi[n - 1] - thetar[n - 1])
                                    / Dsnw[n - 1]
                                )
                            if thetaw[n] > thetar[n]:
                                b[n] = (
                                    1.0 / dth
                                    + 3.0 * ksat[n]
                                    * (thetaw[n] - thetar[n]) ** 2
                                    / cube(phi[n] - thetar[n]) / Dsnw[n]
                                )
                                Qw[n + 1] = ksat[n] * cube(
                                    (thetaw[n] - thetar[n])
                                    / (phi[n] - thetar[n])
                                )
                            rhs[n] = (
                                (thetaw[n] - theta0[n]) / dth
                                + (Qw[n + 1] - Qw[n]) / Dsnw[n]
                            )
                        dtheta[0] = -rhs[0] / b[0]
                        for n in range(1, Nsnow):
                            dtheta[n] = -(
                                a[n] * dtheta[n - 1] + rhs[n]
                            ) / b[n]
                        for n in range(Nsnow):
                            thetaw[n] = max(thetaw[n] + dtheta[n], 0.0)
                            if thetaw[n] > phi[n]:
                                Qw[n + 1] = Qw[n + 1] + (
                                    thetaw[n] - phi[n]
                                ) * Dsnw[n] / dth
                                thetaw[n] = phi[n]
                    for n in range(nsmax):
                        Wflx[n] = Wflx[n] + RHO_WAT * Qw[n] / 10.0
                    Roff = Roff + RHO_WAT * Qw[Nsnow] / 10.0
                for n in range(nsmax):
                    Sliq[n] = RHO_WAT * Dsnw[n] * thetaw[n]
                for n in range(Nsnow):
                    csnow[n] = Sice[n] * HCAP_ICE + Sliq[n] * HCAP_WAT
                    coldcont = csnow[n] * (TM - Tsnow[n])
                    if coldcont > 0.0:
                        dSice = min(Sliq[n], coldcont / LF)
                        Sliq[n] = Sliq[n] - dSice
                        Sice[n] = Sice[n] + dSice
                        Tsnow[n] = Tsnow[n] + LF * dSice / csnow[n]

    snw = array_sum(Sice) + array_sum(Sliq)

    return Nsnow, Gsoil, Roff, snd, snw


@njit(cache=True, fastmath=True)
def soil(params, csoil, dt, Gsoil, ksoil, Tsoil):
    """Update soil temperatures (translation of ``SOIL.F90``)."""
    nsoil = params.nsoil
    a = np.zeros(nsoil)
    b = np.zeros(nsoil)
    c = np.zeros(nsoil)
    rhs = np.zeros(nsoil)
    U = np.zeros(nsoil)
    dTs = np.zeros(nsoil)

    U[0] = 2.0 / (
        params.dzsoil[0] / ksoil[0]
        + params.dzsoil[1] / ksoil[1]
    )
    a[0] = 0.0
    b[0] = csoil[0] + U[0] * dt
    c[0] = -U[0] * dt
    rhs[0] = (Gsoil - U[0] * (Tsoil[0] - Tsoil[1])) * dt
    for n in range(1, nsoil - 1):
        U[n] = 2.0 / (
            params.dzsoil[n] / ksoil[n]
            + params.dzsoil[n + 1] / ksoil[n + 1]
        )
        a[n] = c[n - 1]
        b[n] = csoil[n] + (U[n - 1] + U[n]) * dt
        c[n] = -U[n] * dt
        rhs[n] = (
            U[n - 1] * (Tsoil[n - 1] - Tsoil[n]) * dt
            + U[n] * (Tsoil[n + 1] - Tsoil[n]) * dt
        )
    n = nsoil - 1
    U[n] = ksoil[n] / params.dzsoil[n]
    a[n] = c[n - 1]
    b[n] = csoil[n] + (U[n - 1] + U[n]) * dt
    c[n] = 0.0
    rhs[n] = U[n - 1] * (Tsoil[n - 1] - Tsoil[n]) * dt
    dTs = tridiag(nsoil, a, b, c, rhs)
    for n in range(nsoil):
        Tsoil[n] = Tsoil[n] + dTs[n]


