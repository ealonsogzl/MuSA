"""Numerical helpers, canopy properties and shortwave radiation."""

import math

import numpy as np
from numba import njit

from ._constants import (
    EPS,
    EPSR,
    E0,
    G,
    HCAP_ICE,
    HCAP_WAT,
    HCON_AIR,
    HCON_CLAY,
    HCON_ICE,
    HCON_SAND,
    HCON_WAT,
    LF,
    PI,
    RHO_ICE,
    RHO_WAT,
    SB,
    TM,
)


@njit(cache=True, fastmath=True)
def qsat(P, T):
    """Saturation specific humidity (same as ``QSAT.F90``)."""
    Tc = T - TM
    if Tc > 0.0:
        es = E0 * math.exp(17.5043 * Tc / (241.3 + Tc))
    else:
        es = E0 * math.exp(22.4422 * Tc / (272.186 + Tc))
    return EPS * es / P


@njit(cache=True, fastmath=True)
def psim(z, rL):
    """Stability function for momentum (same as ``PSIMH.F90``)."""
    zeta = z * rL
    if zeta > 1.0:
        zeta = 1.0
    if zeta < -2.0:
        zeta = -2.0
    if zeta > 0.0:
        return -5.0 * zeta
    x = (1.0 - 16.0 * zeta) ** 0.25
    return (
        2.0 * math.log((1.0 + x) / 2.0)
        + math.log((1.0 + x * x) / 2.0)
        - 2.0 * math.atan(x)
        + PI / 2.0
    )


@njit(cache=True, fastmath=True)
def psih(z, rL):
    """Stability function for heat (same as ``PSIMH.F90``)."""
    zeta = z * rL
    if zeta > 1.0:
        zeta = 1.0
    if zeta < -2.0:
        zeta = -2.0
    if zeta > 0.0:
        return -5.0 * zeta
    x = (1.0 - 16.0 * zeta) ** 0.25
    return 2.0 * math.log((1.0 + x * x) / 2.0)


@njit(cache=True, fastmath=True)
def ludcmp(A, b):
    """Solve ``A x = b`` with the same scaled partial pivoting as LUDCMP.F90."""
    n = A.shape[0]
    Acp = A.copy()
    x = b.copy()
    vv = np.zeros(n)
    indx = np.zeros(n, dtype=np.int64)

    for i in range(n):
        aamax = 0.0
        for j in range(n):
            if abs(Acp[i, j]) > aamax:
                aamax = abs(Acp[i, j])
        if aamax == 0.0:
            return x
        vv[i] = 1.0 / aamax

    for j in range(n):
        for i in range(j):
            s = Acp[i, j]
            if i > 0:
                for k in range(i):
                    s -= Acp[i, k] * Acp[k, j]
                Acp[i, j] = s

        aamax = 0.0
        imax = j
        for i in range(j, n):
            s = Acp[i, j]
            for k in range(j):
                s -= Acp[i, k] * Acp[k, j]
            Acp[i, j] = s
            dum = vv[i] * abs(s)
            if dum >= aamax:
                imax = i
                aamax = dum

        if j != imax:
            for k in range(n):
                dum = Acp[imax, k]
                Acp[imax, k] = Acp[j, k]
                Acp[j, k] = dum
            vv[imax] = vv[j]

        indx[j] = imax
        if Acp[j, j] == 0.0:
            Acp[j, j] = 1.0e-20
        if j != n - 1:
            dum = 1.0 / Acp[j, j]
            for i in range(j + 1, n):
                Acp[i, j] *= dum

    ii = 0
    for i in range(n):
        ll = indx[i]
        s = x[ll]
        x[ll] = x[i]
        if ii != 0:
            for j in range(ii - 1, i):
                s -= Acp[i, j] * x[j]
        elif s != 0.0:
            ii = i + 1
        x[i] = s

    for i in range(n - 1, -1, -1):
        s = x[i]
        for j in range(i + 1, n):
            s -= Acp[i, j] * x[j]
        x[i] = s / Acp[i, i]

    return x


@njit(cache=True, fastmath=True)
def tridiag(n, a, b, c, r):
    """Solve a tridiagonal system (same as TRIDIAG.F90)."""
    x = np.empty(n)
    g = np.empty(n)
    beta = b[0]
    x[0] = r[0] / beta
    for i in range(1, n):
        g[i] = c[i - 1] / beta
        beta = b[i] - a[i] * g[i]
        x[i] = (r[i] - a[i] * x[i - 1]) / beta
    for i in range(n - 2, -1, -1):
        x[i] -= g[i + 1] * x[i + 1]
    return x


@njit(cache=True, fastmath=True)
def twostream(params, elev, fcans, lveg):
    """Two-stream reflectance and transmittance of one canopy layer."""
    omega = (1.0 - fcans) * params.avg0 + fcans * params.avgs
    beta = 0.67
    g1 = 2.0 * (1.0 - (1.0 - beta) * omega)
    g2 = 2.0 * beta * omega
    k = math.sqrt(g1 * g1 - g2 * g2)
    tau = params.kext * lveg

    # Diffuse radiation
    D = k + g1 + (k - g1) * math.exp(-2.0 * k * tau)
    rdif = (g2 / D) * (1.0 - math.exp(-2.0 * k * tau))
    tdif = 2.0 * (k / D) * math.exp(-k * tau)

    # Direct beam
    fdir = 0.0
    rdir = 0.0
    tdir = 0.0
    if elev > 0.0:
        mu = math.sin(elev)
        beta0 = (0.5 + mu) * (1.0 - mu * math.log((1.0 + mu) / mu))
        g3 = beta0
        g4 = 1.0 - beta0
        a1 = g1 * g4 + g2 * g3
        a2 = g1 * g3 + g2 * g4
        D = (1.0 - k * k * mu * mu) * (
            (k + g1) * math.exp(k * tau) + (k - g1) * math.exp(-k * tau)
        )
        b1 = (1.0 - k * mu) * (a2 + k * g3) * math.exp(k * tau)
        b2 = (1.0 + k * mu) * (a2 - k * g3) * math.exp(-k * tau)
        b3 = 2.0 * k * (g3 - a2 * mu) * math.exp(-tau / mu)
        rdir = (omega / D) * (b1 - b2 - b3)

        b1 = (1.0 + k * mu) * (a1 + k * g4) * math.exp(k * tau)
        b2 = (1.0 - k * mu) * (a1 - k * g4) * math.exp(-k * tau)
        b3 = 2.0 * k * (g4 + a1 * mu) * math.exp(tau / mu)
        tdir = math.exp(-tau / mu)
        if tau > 30.0 * mu:
            fdir = 2.0 * k * (omega / D) * (g4 + a1 * mu)
        else:
            fdir = math.exp(-tau / mu) * (omega / D) * (b2 + b3 - b1)

    return fdir, rdif, rdir, tdif, tdir


@njit(cache=True, fastmath=True)
def canopy(params, Sveg, Tveg, VAI, cveg, fcans, lveg, Scap, Tveg0):
    """Properties of the canopy layers (same as CANOPY.F90)."""
    ncnpy = params.ncnpy
    if params.canmod == 1:
        lveg[0] = VAI
    else:
        lveg[0] = params.fvg1 * VAI
        lveg[1] = (1.0 - params.fvg1) * VAI

    for n in range(ncnpy):
        cveg[n] = params.cvai * lveg[n] + HCAP_ICE * Sveg[n]
        fcans[n] = 0.0
        Scap[n] = 0.0
        Tveg0[n] = Tveg[n]

    if VAI > 0.0:
        for n in range(ncnpy):
            Scap[n] = params.svai * lveg[n]
            if params.svai > 0.0:
                fcans[n] = (Sveg[n] / Scap[n]) ** 0.67
                if fcans[n] > 1.0:
                    fcans[n] = 1.0


@njit(cache=True, fastmath=True)
def swrad(params, alb0, Dsnw, dt, elev, fcans, lveg, Sdif, Sdir, Sf, Tsrf,
          albs, SWveg, tdif):
    """Surface and canopy net shortwave radiation (same as SWRAD.F90)."""
    # Prognostic or diagnostic snow albedo
    if params.albedo == 1:
        albs = params.asmn + (params.asmx - params.asmn) * (Tsrf - TM) / params.Talb
    else:
        tdec = params.tcld
        if Tsrf >= TM:
            tdec = params.tmlt
        alim = (
            params.asmn / tdec + params.asmx * Sf / params.Salb
        ) / (1.0 / tdec + Sf / params.Salb)
        albs = alim + (albs - alim) * math.exp(-(1.0 / tdec + Sf / params.Salb) * dt)

    if albs > params.asmx:
        albs = params.asmx
    if albs < params.asmn:
        albs = params.asmn

    snd = 0.0
    for n in range(params.nsmax):
        snd += Dsnw[n]

    if params.snfrac == 1:
        fsnow = snd / params.hfsn
        if fsnow > 1.0:
            fsnow = 1.0
    elif params.snfrac == 2:
        fsnow = math.tanh(snd / params.hfsn)
    else:
        fsnow = snd / (snd + params.hfsn)

    asrf = (1.0 - fsnow) * alb0 + fsnow * albs
    for n in range(params.ncnpy):
        SWveg[n] = 0.0
        tdif[n] = 0.0

    SWout = asrf * (Sdif + Sdir)
    SWsub = Sdif + Sdir

    if lveg[0] > 0.0:
        acan = np.empty(params.ncnpy)
        fdir = np.zeros(params.ncnpy)
        rdif = np.zeros(params.ncnpy)
        rdir = np.zeros(params.ncnpy)
        tdir = np.zeros(params.ncnpy)

        if params.canrad == 1:
            for n in range(params.ncnpy):
                acan[n] = (1.0 - fcans[n]) * params.acn0 + fcans[n] * params.acns
                tdif[n] = math.exp(-1.6 * params.kext * lveg[n])
                tdir[n] = tdif[n]
                if elev > 0.0:
                    tdir[n] = math.exp(-params.kext * lveg[n] / math.sin(elev))
                rdif[n] = (1.0 - tdif[n]) * acan[n]
                rdir[n] = (1.0 - tdir[n]) * acan[n]
        else:
            for n in range(params.ncnpy):
                fdir[n], rdif[n], rdir[n], tdif[n], tdir[n] = twostream(
                    params, elev, fcans[n], lveg[n]
                )

        if params.canmod == 1:
            A = np.eye(3)
            b = np.zeros(3)
            A[0, 1] = -rdif[0]
            A[1, 0] = -asrf
            A[2, 1] = -tdif[0]
            b[0] = tdif[0] * Sdif + fdir[0] * Sdir
            b[1] = asrf * tdir[0] * Sdir
            b[2] = rdif[0] * Sdif + rdir[0] * Sdir
            x = ludcmp(A, b)
            SWout = x[2]
            SWveg[0] = (
                Sdif - x[0] + x[1] - x[2] + (1.0 - tdir[0]) * Sdir
            )
            SWsub = x[0] + tdir[0] * Sdir
        else:
            A = np.eye(5)
            b = np.zeros(5)
            A[0, 3] = -rdif[0]
            A[1, 0] = -tdif[1]
            A[1, 2] = -rdif[1]
            A[2, 1] = -asrf
            A[3, 0] = -rdif[1]
            A[3, 2] = -tdif[1]
            A[4, 3] = -tdif[0]
            b[0] = tdif[0] * Sdif + fdir[0] * Sdir
            b[1] = fdir[1] * tdir[0] * Sdir
            b[2] = asrf * tdir[0] * tdir[1] * Sdir
            b[3] = rdir[1] * tdir[0] * Sdir
            b[4] = rdif[0] * Sdif + rdir[0] * Sdir
            x = ludcmp(A, b)
            SWout = x[4]
            SWveg[0] = (
                Sdif - x[0] + x[3] - x[4] + (1.0 - tdir[0]) * Sdir
            )
            SWveg[1] = (
                x[0] - x[1] + x[2] - x[3]
                + tdir[0] * (1.0 - tdir[1]) * Sdir
            )
            SWsub = x[1] + tdir[0] * tdir[1] * Sdir

    SWsrf = (1.0 - asrf) * SWsub

    # Diffuse transmittance for longwave radiation
    for n in range(params.ncnpy):
        tdif[n] = math.exp(-1.6 * params.kext * lveg[n])

    return albs, fsnow, SWout, SWsrf, SWsub


@njit(cache=True, fastmath=True)
def thermal(params, Dsnw, Nsnow, Sice, Sliq, Tsnow, Tsoil, Vsmc, csoil,
            ksnow, ksoil):
    """Thermal properties of snow and soil (same as THERMAL.F90)."""
    nsmax = params.nsmax
    nsoil = params.nsoil

    for n in range(nsmax):
        ksnow[n] = params.kfix

    if params.condct == 1:
        for n in range(Nsnow):
            rhos = params.rhof
            if params.densty != 0 and Dsnw[n] > EPSR:
                rhos = (Sice[n] + Sliq[n]) / Dsnw[n]
            ksnow[n] = 2.224 * (rhos / RHO_WAT) ** 1.885

    dPsidT = -RHO_ICE * LF / (RHO_WAT * G * TM)
    gs1 = 0.0
    for n in range(nsoil):
        csoil[n] = params.hcap_soil * params.dzsoil[n]
        ksoil[n] = params.hcon_soil
        if Vsmc[n] > EPSR:
            dthudT = 0.0
            sthu = Vsmc[n]
            sthf = 0.0
            Tc = Tsoil[n] - TM
            Tmax = TM + (params.sathh / dPsidT) * (params.Vsat / Vsmc[n]) ** params.b
            if Tsoil[n] < Tmax:
                dthudT = (
                    (-dPsidT * params.Vsat / (params.b * params.sathh))
                    * (dPsidT * Tc / params.sathh) ** (-1.0 / params.b - 1.0)
                )
                sthu = params.Vsat * (dPsidT * Tc / params.sathh) ** (-1.0 / params.b)
                if sthu > Vsmc[n]:
                    sthu = Vsmc[n]
                sthf = (Vsmc[n] - sthu) * RHO_WAT / RHO_ICE

            Mf = RHO_ICE * params.dzsoil[n] * sthf
            Mu = RHO_WAT * params.dzsoil[n] * sthu
            csoil[n] = (
                params.hcap_soil * params.dzsoil[n]
                + HCAP_ICE * Mf
                + HCAP_WAT * Mu
                + RHO_WAT * params.dzsoil[n]
                * ((HCAP_WAT - HCAP_ICE) * Tc + LF) * dthudT
            )

            Smf = RHO_ICE * sthf / (RHO_WAT * params.Vsat)
            Smu = sthu / params.Vsat
            thice = 0.0
            if Smf > 0.0:
                thice = params.Vsat * Smf / (Smu + Smf)
            thwat = 0.0
            if Smu > 0.0:
                thwat = params.Vsat * Smu / (Smu + Smf)
            hcon_sat = (
                params.hcon_soil
                * (HCON_WAT ** thwat)
                * (HCON_ICE ** thice)
                / (HCON_AIR ** params.Vsat)
            )
            ksoil[n] = (hcon_sat - params.hcon_soil) * (Smf + Smu) + params.hcon_soil

            if n == 0:
                gs1 = params.gsat * max((Smu * params.Vsat / params.Vcrit) ** 2, 1.0)

    Ds1 = max(params.dzsoil[0], Dsnw[0])
    Ts1 = Tsoil[0] + (Tsnow[0] - Tsoil[0]) * Dsnw[0] / params.dzsoil[0]
    ks1 = params.dzsoil[0] / (
        2.0 * Dsnw[0] / ksnow[0]
        + (params.dzsoil[0] - 2.0 * Dsnw[0]) / ksoil[0]
    )
    snd = 0.0
    for n in range(nsmax):
        snd += Dsnw[n]
    if snd > 0.5 * params.dzsoil[0]:
        ks1 = ksnow[0]
    if snd > params.dzsoil[0]:
        Ts1 = Tsnow[0]

    return Ds1, gs1, ks1, Ts1
