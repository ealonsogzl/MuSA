"""Surface and canopy energy balance (translation of ``SRFEBAL.F90``)."""

import math

import numpy as np
from numba import njit

from ._constants import CP, G, LF, LS, LV, RAIR, RWAT, SB, TM, VKMAN
from ._numeric import array_sum, cube, fourth
from ._radiation import ludcmp, psih, psim, qsat


@njit(cache=True, fastmath=True)
def srfb(params, cveg, Ds1, dt, fcans, fsnow, gs1, ks1, lveg, LW, Ps, Qa,
         SWsrf, Sveg, SWveg, Ta, tdif, Ts1, Tveg0, Ua, VAI, vegh,
         zT, zU, Tsrf, Qcan, Sice, Tcan, Tveg, Eveg):
    """Surface and canopy energy balance.

    Scalar state variables (``Tsrf``) are returned; array state variables
    (``Qcan``, ``Tcan``, ``Tveg``, ``Eveg``) are updated in place.
    """
    ncnpy = params.ncnpy

    # Work arrays
    zh = np.zeros(ncnpy)
    gv = np.zeros(ncnpy)
    wveg = np.zeros(ncnpy)
    Qveg = np.zeros(ncnpy)
    Dveg = np.zeros(ncnpy)
    Lcan = np.zeros(ncnpy)
    Hveg = np.zeros(ncnpy)
    Rveg = np.zeros(ncnpy)
    dEv = np.zeros(ncnpy)
    dHv = np.zeros(ncnpy)
    dQc = np.zeros(ncnpy)
    dTc = np.zeros(ncnpy)
    dTv = np.zeros(ncnpy)

    J = np.zeros((3 * ncnpy + 1, 3 * ncnpy + 1))
    f = np.zeros(3 * ncnpy + 1)

    # Defaults (also make Numba's type inference unambiguous)
    d = 0.0
    fveg = 0.0
    z0g = 0.0
    z0h = 0.0
    z0v = 0.0
    rL = 0.0
    ustar = 0.0
    ga = 0.0
    gs = 0.0
    gc = 0.0
    Kh = 0.0
    rd = 0.0
    ro = 0.0
    Uh = 0.0
    Uc = 0.0
    Esrf = 0.0
    Gsrf = 0.0
    H = 0.0
    LE = 0.0
    LWout = 0.0
    LWsub = 0.0
    Melt = 0.0
    subl = 0.0
    Tsub = 0.0
    Usub = 0.0
    Hsrf = 0.0
    E = 0.0
    Ecan = 0.0
    Hcan = 0.0
    ebal = 0.0
    dTs = 0.0
    dEs = 0.0
    dGs = 0.0
    dHs = 0.0
    wsrf = 0.0
    Rsrf = 0.0
    Qsrf = 0.0
    Dsrf = 0.0
    rho = 0.0
    Lsrf = LS
    Ssub = 0.0

    for n in range(ncnpy):
        Eveg[n] = 0.0
        Hveg[n] = 0.0
        Lcan[n] = 0.0

    # Measurement heights (ZOFFST option)
    if params.zoffst == 0:
        zU1 = zU
        zT1 = zT
    else:
        zU1 = zU + vegh
        zT1 = zT + vegh

    # Canopy layer heights
    if params.canmod == 1:
        zh[0] = params.hbas + 0.5 * (vegh - params.hbas)
    else:
        zh[0] = (1.0 - 0.5 * params.fvg1) * vegh
        zh[1] = 0.5 * (1.0 - params.fvg1) * vegh

    # Roughness lengths
    fveg = 1.0 - math.exp(-params.kext * VAI)
    d = 0.67 * vegh
    z0g = (params.z0sn ** fsnow) * (params.z0sf ** (1.0 - fsnow))
    z0h = 0.1 * z0g
    z0v = 0.1 * vegh

    # Saturation humidity and air density
    Qsrf = qsat(Ps, Tsrf)
    Lsrf = LS
    if Tsrf > TM:
        Lsrf = LV
    Dsrf = Lsrf * Qsrf / (RWAT * Tsrf ** 2)
    rho = Ps / (RAIR * Ta)

    if VAI == 0.0:
        # ---------------------------------------------------------------
        # Open ground
        # ---------------------------------------------------------------
        for n in range(ncnpy):
            Eveg[n] = 0.0
            Hveg[n] = 0.0
            Lcan[n] = 0.0
        LWsub = LW
        rL = 0.0
        ustar = VKMAN * Ua / math.log(zU1 / z0g)
        ga = VKMAN * ustar / math.log(zT1 / z0h)

        for ne in range(1, 11):
            if params.exchng == 1:
                if ne < 8:
                    rL = -VKMAN * G * ga * (Tsrf - Ta) / (Ta * cube(ustar))
                ustar = VKMAN * Ua / (
                    math.log(zU1 / z0g) - psim(zU1, rL) + psim(z0g, rL)
                )
                ga = VKMAN * ustar / (
                    math.log(zT1 / z0h) - psih(zT1, rL) + psih(z0h, rL)
                )

            # Surface water availability
            if Qa > Qsrf:
                wsrf = 1.0
            else:
                wsrf = fsnow + (1.0 - fsnow) * gs1 / (gs1 + ga)

            # Explicit fluxes
            Esrf = rho * wsrf * ga * (Qsrf - Qa)
            for n in range(ncnpy):
                Eveg[n] = 0.0
            Gsrf = 2.0 * ks1 * (Tsrf - Ts1) / Ds1
            Hsrf = CP * rho * ga * (Tsrf - Ta)
            for n in range(ncnpy):
                Hveg[n] = 0.0
            Melt = 0.0
            Rsrf = SWsrf + LW - SB * fourth(Tsrf)

            # Surface energy balance increments without melt
            dTs = (Rsrf - Gsrf - Hsrf - Lsrf * Esrf) / (
                4.0 * SB * cube(Tsrf)
                + 2.0 * ks1 / Ds1
                + rho * (CP + Lsrf * Dsrf * wsrf) * ga
            )
            dEs = rho * wsrf * ga * Dsrf * dTs
            dGs = 2.0 * ks1 * dTs / Ds1
            dHs = CP * rho * ga * dTs

            # Surface melting
            if Tsrf + dTs > TM and Sice[0] > 0.0:
                Melt = array_sum(Sice) / dt
                dTs = (Rsrf - Gsrf - Hsrf - Lsrf * Esrf - LF * Melt) / (
                    4.0 * SB * cube(Tsrf)
                    + 2.0 * ks1 / Ds1
                    + rho * (CP + LS * Dsrf * wsrf) * ga
                )
                dEs = rho * wsrf * ga * Dsrf * dTs
                dGs = 2.0 * ks1 * dTs / Ds1
                dHs = CP * rho * ga * dTs
                if Tsrf + dTs < TM:
                    Qsrf = qsat(Ps, TM)
                    Esrf = rho * wsrf * ga * (Qsrf - Qa)
                    Gsrf = 2.0 * ks1 * (TM - Ts1) / Ds1
                    Hsrf = CP * rho * ga * (TM - Ta)
                    Rsrf = SWsrf + LW - SB * fourth(TM)
                    Melt = (Rsrf - Gsrf - Hsrf - Lsrf * Esrf) / LF
                    if Melt < 0.0:
                        Melt = 0.0
                    dEs = 0.0
                    dGs = 0.0
                    dHs = 0.0
                    dTs = TM - Tsrf

            # Update surface temperature and fluxes
            Esrf = Esrf + dEs
            Gsrf = Gsrf + dGs
            Hsrf = Hsrf + dHs
            Tsrf = Tsrf + dTs
            ebal = (
                SWsrf + LW - SB * fourth(Tsrf) - Gsrf - Hsrf
                - Lsrf * Esrf - LF * Melt
            )
            if ne > 4 and abs(ebal) < 0.01:
                break

    else:
        # ---------------------------------------------------------------
        # Forest
        # ---------------------------------------------------------------
        rL = 0.0
        ustar = (
            fveg * VKMAN * Ua / math.log((zU1 - d) / z0v)
            + (1.0 - fveg) * VKMAN * Ua / math.log(zU1 / z0g)
        )
        Kh = VKMAN * ustar * (vegh - d)
        rd = (
            math.log((zT1 - d) / (vegh - d)) / (VKMAN * ustar)
            + vegh * (math.exp(params.wcan * (1.0 - zh[0] / vegh)) - 1.0)
            / (params.wcan * Kh)
        )
        ro = math.log(zT1 / zh[0]) / (VKMAN * ustar)
        ga = fveg / rd + (1.0 - fveg) / ro

        for ne in range(1, 11):
            if params.exchng == 1:
                ustar = (
                    fveg * VKMAN * Ua
                    / (
                        math.log((zU1 - d) / z0v)
                        - psim(zU1 - d, rL) + psim(z0v, rL)
                    )
                    + (1.0 - fveg) * VKMAN * Ua
                    / (
                        math.log(zU1 / z0g)
                        - psim(zU1, rL) + psim(z0g, rL)
                    )
                )
                if ne < 8:
                    rL = -VKMAN * G * ga * (Tcan[0] - Ta) / (Ta * cube(ustar))
                if rL > 0.0:
                    Kh = VKMAN * ustar * (vegh - d) / (
                        1.0 + 5.0 * (vegh - d) * rL
                    )
                else:
                    Kh = VKMAN * ustar * (vegh - d) * math.sqrt(
                        1.0 - 16.0 * (vegh - d) * rL
                    )
                rd = (
                    (
                        math.log((zT1 - d) / (vegh - d))
                        - psih(zT1 - d, rL) + psih(vegh - d, rL)
                    )
                    / (VKMAN * ustar)
                    + vegh
                    * (math.exp(params.wcan * (1.0 - zh[0] / vegh)) - 1.0)
                    / (params.wcan * Kh)
                )
                ro = (
                    math.log(zT1 / zh[0])
                    - psih(zT1, rL) + psih(zh[0], rL)
                ) / (VKMAN * ustar)
                ga = fveg / rd + (1.0 - fveg) / ro

            Uh = (ustar / VKMAN) * (
                math.log((vegh - d) / z0v)
                - psim(vegh - d, rL) + psim(z0v, rL)
            )
            for n in range(ncnpy):
                Uc = (
                    fveg * math.exp(params.wcan * (zh[n] / vegh - 1.0)) * Uh
                    + (1.0 - fveg) * (ustar / VKMAN)
                    * (
                        math.log(zh[n] / z0g)
                        - psim(zh[n], rL) + psim(z0g, rL)
                    )
                )
                gv[n] = math.sqrt(Uc) * lveg[n] / params.leaf

            if params.canmod == 2:
                rd = (
                    vegh * math.exp(params.wcan)
                    * (
                        math.exp(-params.wcan * zh[1] / vegh)
                        - math.exp(-params.wcan * zh[0] / vegh)
                    )
                    / (params.wcan * Kh)
                )
                ro = (
                    math.log(zh[0] / zh[1])
                    - psih(zh[0], rL) + psih(zh[1], rL)
                ) / (VKMAN * ustar)
                gc = fveg / rd + (1.0 - fveg) / ro

            n = ncnpy - 1
            Uc = math.exp(params.wcan * (params.hbas / vegh - 1.0)) * Uh
            rd = (
                math.log(params.hbas / z0g)
                * math.log(params.hbas / z0h)
                / (VKMAN ** 2 * Uc)
                + vegh * math.exp(params.wcan)
                * (
                    math.exp(-params.wcan * params.hbas / vegh)
                    - math.exp(-params.wcan * zh[n] / vegh)
                )
                / (params.wcan * Kh)
            )
            ro = (
                math.log(zh[n] / z0h)
                - psih(zh[n], rL) + psih(z0h, rL)
            ) / (VKMAN * ustar)
            gs = fveg / rd + (1.0 - fveg) / ro

            # Saturation humidity
            for nn in range(ncnpy):
                Qveg[nn] = qsat(Ps, Tveg[nn])
                Lcan[nn] = LS
                if Tveg[nn] > TM:
                    Lcan[nn] = LV
                Dveg[nn] = Lcan[nn] * Qveg[nn] / (RWAT * Tveg[nn] ** 2)

            # Water availability
            if Qcan[ncnpy - 1] > Qsrf:
                wsrf = 1.0
            else:
                wsrf = fsnow + (1.0 - fsnow) * gs1 / (gs1 + gs)
            for nn in range(ncnpy):
                if Qcan[nn] > Qveg[nn]:
                    wveg[nn] = 1.0
                else:
                    wveg[nn] = fcans[nn] + (1.0 - fcans[nn]) * params.gsnf / (
                        params.gsnf + gv[nn]
                    )

            if params.canmod == 1:
                # -----------------------------------------------------------
                # One-layer canopy model
                # -----------------------------------------------------------
                # Explicit fluxes
                E = rho * ga * (Qcan[0] - Qa)
                Esrf = rho * wsrf * gs * (Qsrf - Qcan[0])
                Eveg[0] = rho * wveg[0] * gv[0] * (Qveg[0] - Qcan[0])
                Gsrf = 2.0 * ks1 * (Tsrf - Ts1) / Ds1
                H = rho * CP * ga * (Tcan[0] - Ta)
                Hsrf = rho * CP * gs * (Tsrf - Tcan[0])
                Hveg[0] = rho * CP * gv[0] * (Tveg[0] - Tcan[0])
                Melt = 0.0
                Rsrf = (
                    SWsrf + tdif[0] * LW - SB * fourth(Tsrf)
                    + (1.0 - tdif[0]) * SB * fourth(Tveg[0])
                )
                Rveg[0] = (
                    SWveg[0] + (1.0 - tdif[0])
                    * (LW + SB * fourth(Tsrf) - 2.0 * SB * fourth(Tveg[0]))
                )
    
                # Surface energy balance increments without melt
                J[0, 0] = (
                    -rho * gs * (CP + Lsrf * Dsrf * wsrf)
                    - 4.0 * SB * cube(Tsrf) - 2.0 * ks1 / Ds1
                )
                J[0, 1] = Lsrf * rho * wsrf * gs
                J[0, 2] = rho * CP * gs
                J[0, 3] = 4.0 * (1.0 - tdif[0]) * SB * cube(Tveg[0])
                J[1, 0] = 4.0 * (1.0 - tdif[0]) * SB * cube(Tsrf)
                J[1, 1] = Lcan[0] * rho * wveg[0] * gv[0]
                J[1, 2] = rho * CP * gv[0]
                J[1, 3] = (
                    -rho * gv[0] * (CP + Lcan[0] * Dveg[0] * wveg[0])
                    - 8.0 * (1.0 - tdif[0]) * SB * cube(Tveg[0])
                    - cveg[0] / dt
                )
                J[2, 0] = -gs
                J[2, 1] = 0.0
                J[2, 2] = ga + gs + gv[0]
                J[2, 3] = -gv[0]
                J[3, 0] = -Dsrf * wsrf * gs
                J[3, 1] = ga + wsrf * gs + wveg[0] * gv[0]
                J[3, 2] = 0.0
                J[3, 3] = -Dveg[0] * wveg[0] * gv[0]
                f[0] = -(Rsrf - Gsrf - Hsrf - Lsrf * Esrf)
                f[1] = -(
                    Rveg[0] - Hveg[0] - Lcan[0] * Eveg[0]
                    - cveg[0] * (Tveg[0] - Tveg0[0]) / dt
                )
                f[2] = -(H - Hveg[0] - Hsrf) / (rho * CP)
                f[3] = -(E - Eveg[0] - Esrf) / rho
                x = ludcmp(J, f)
                dTs = x[0]
                dQc[0] = x[1]
                dTc[0] = x[2]
                dTv[0] = x[3]
                dEs = rho * wsrf * gs * (Dsrf * dTs - dQc[0])
                dEv[0] = rho * wveg[0] * gv[0] * (Dveg[0] * dTv[0] - dQc[0])
                dGs = 2.0 * ks1 * dTs / Ds1
                dHs = rho * CP * gs * (dTs - dTc[0])
                dHv[0] = rho * CP * gv[0] * (dTv[0] - dTc[0])
    
                # Surface melting
                if Tsrf + dTs > TM and Sice[0] > 0.0:
                    Melt = array_sum(Sice) / dt
                    f[0] = f[0] + LF * Melt
                    x = ludcmp(J, f)
                    dTs = x[0]
                    dQc[0] = x[1]
                    dTc[0] = x[2]
                    dTv[0] = x[3]
                    dEs = rho * wsrf * gs * (Dsrf * dTs - dQc[0])
                    dEv[0] = rho * wveg[0] * gv[0] * (Dveg[0] * dTv[0] - dQc[0])
                    dGs = 2.0 * ks1 * dTs / Ds1
                    dHs = rho * CP * gs * (dTs - dTc[0])
                    dHv[0] = rho * CP * gv[0] * (dTv[0] - dTc[0])
                    if Tsrf + dTs < TM:
                        Qsrf = qsat(Ps, TM)
                        Esrf = rho * wsrf * gs * (Qsrf - Qcan[0])
                        Gsrf = 2.0 * ks1 * (TM - Ts1) / Ds1
                        Hsrf = rho * CP * gs * (TM - Tcan[0])
                        Rsrf = (
                            SWsrf + tdif[0] * LW - SB * fourth(TM)
                            + (1.0 - tdif[0]) * SB * fourth(Tveg[0])
                        )
                        Rveg[0] = (
                            SWveg[0] + (1.0 - tdif[0])
                            * (LW + SB * fourth(TM) - 2.0 * SB * fourth(Tveg[0]))
                        )
                        J[0, 0] = -1.0
                        J[1, 0] = 0.0
                        J[2, 0] = 0.0
                        J[3, 0] = 0.0
                        f[0] = -(Rsrf - Gsrf - Hsrf - Lsrf * Esrf)
                        f[1] = -(
                            Rveg[0] - Hveg[0] - Lcan[0] * Eveg[0]
                            - cveg[0] * (Tveg[0] - Tveg0[0]) / dt
                        )
                        f[2] = -(H - Hveg[0] - Hsrf) / (rho * CP)
                        f[3] = -(E - Eveg[0] - Esrf) / rho
                        x = ludcmp(J, f)
                        Melt = x[0] / LF
                        dQc[0] = x[1]
                        dTc[0] = x[2]
                        dTv[0] = x[3]
                        dTs = TM - Tsrf
                        dEs = 0.0
                        dEv[0] = rho * wveg[0] * gv[0] * (
                            Dveg[0] * dTv[0] - dQc[0]
                        )
                        dGs = 0.0
                        dHs = 0.0
                        dHv[0] = rho * CP * gv[0] * (dTv[0] - dTc[0])
    
                LWout = (
                    (1.0 - tdif[0]) * SB * fourth(Tveg[0])
                    + tdif[0] * SB * fourth(Tsrf)
                )
                LWsub = tdif[0] * LW + (1.0 - tdif[0]) * SB * fourth(Tveg[0])
    
            else:
                # -----------------------------------------------------------
                # Two-layer canopy model
                # -----------------------------------------------------------
                # Explicit fluxes
                E = rho * ga * (Qcan[0] - Qa)
                Ecan = rho * gc * (Qcan[1] - Qcan[0])
                Esrf = rho * wsrf * gs * (Qsrf - Qcan[1])
                Eveg[0] = rho * wveg[0] * gv[0] * (Qveg[0] - Qcan[0])
                Eveg[1] = rho * wveg[1] * gv[1] * (Qveg[1] - Qcan[1])
                Gsrf = 2.0 * ks1 * (Tsrf - Ts1) / Ds1
                H = rho * CP * ga * (Tcan[0] - Ta)
                Hcan = rho * CP * gc * (Tcan[1] - Tcan[0])
                Hsrf = rho * CP * gs * (Tsrf - Tcan[1])
                Hveg[0] = rho * CP * gv[0] * (Tveg[0] - Tcan[0])
                Hveg[1] = rho * CP * gv[1] * (Tveg[1] - Tcan[1])
                Melt = 0.0
                Rsrf = (
                    SWsrf + tdif[0] * tdif[1] * LW
                    + (1.0 - tdif[0]) * tdif[1] * SB * fourth(Tveg[0])
                    + (1.0 - tdif[1]) * SB * fourth(Tveg[1])
                    - SB * fourth(Tsrf)
                )
                Rveg[0] = (
                    SWveg[0] + (1.0 - tdif[0])
                    * (
                        LW - 2.0 * SB * fourth(Tveg[0])
                        + (1.0 - tdif[1]) * SB * fourth(Tveg[1])
                        + tdif[1] * SB * fourth(Tsrf)
                    )
                )
                Rveg[1] = (
                    SWveg[1] + (1.0 - tdif[1])
                    * (
                        tdif[0] * LW
                        + (1.0 - tdif[0]) * SB * fourth(Tveg[0])
                        - 2.0 * SB * fourth(Tveg[1])
                        + SB * fourth(Tsrf)
                    )
                )
    
                # Surface energy balance increments without melt
                J[0, 0] = (
                    -rho * gs * (CP + Lsrf * Dsrf * wsrf)
                    - 4.0 * SB * cube(Tsrf) - 2.0 * ks1 / Ds1
                )
                J[0, 1] = 0.0
                J[0, 2] = 0.0
                J[0, 3] = 4.0 * (1.0 - tdif[0]) * tdif[1] * SB * cube(Tveg[0])
                J[0, 4] = Lsrf * rho * wsrf * gs
                J[0, 5] = rho * CP * gs
                J[0, 6] = 4.0 * (1.0 - tdif[1]) * SB * cube(Tveg[1])
                J[1, 0] = 4.0 * (1.0 - tdif[0]) * tdif[1] * SB * cube(Tveg[1])
                J[1, 1] = Lcan[0] * rho * wveg[0] * gv[0]
                J[1, 2] = rho * CP * gv[0]
                J[1, 3] = (
                    -rho * gv[0] * (CP + Lcan[0] * Dveg[0] * wveg[0])
                    - 8.0 * (1.0 - tdif[0]) * SB * cube(Tveg[0])
                    - cveg[0] / dt
                )
                J[1, 4] = 0.0
                J[1, 5] = 0.0
                J[1, 6] = (
                    4.0 * (1.0 - tdif[0]) * (1.0 - tdif[1])
                    * SB * cube(Tveg[1])
                )
                J[2, 0] = 4.0 * (1.0 - tdif[1]) * SB * cube(Tsrf)
                J[2, 1] = 0.0
                J[2, 2] = 0.0
                J[2, 3] = (
                    4.0 * (1.0 - tdif[0]) * (1.0 - tdif[1])
                    * SB * cube(Tveg[0])
                )
                J[2, 4] = Lcan[1] * rho * wveg[1] * gv[1]
                J[2, 5] = rho * CP * gv[1]
                J[2, 6] = (
                    -rho * gv[1] * (CP + Lcan[1] * Dveg[1] * wveg[1])
                    - 8.0 * (1.0 - tdif[1]) * SB * cube(Tveg[1])
                    - cveg[1] / dt
                )
                J[3, 0] = 0.0
                J[3, 1] = 0.0
                J[3, 2] = ga + gc + gv[0]
                J[3, 3] = -gv[0]
                J[3, 4] = 0.0
                J[3, 5] = -gc
                J[3, 6] = 0.0
                J[4, 0] = -gs
                J[4, 1] = 0.0
                J[4, 2] = -gc
                J[4, 3] = 0.0
                J[4, 4] = 0.0
                J[4, 5] = gc + gs + gv[1]
                J[4, 6] = -gv[1]
                J[5, 0] = 0.0
                J[5, 1] = ga + gc + wveg[0] * gv[0]
                J[5, 2] = 0.0
                J[5, 3] = -Dveg[0] * wveg[0] * gv[0]
                J[5, 4] = -gc
                J[5, 5] = 0.0
                J[5, 6] = 0.0
                J[6, 0] = -Dsrf * wsrf * gs
                J[6, 1] = -gc
                J[6, 2] = 0.0
                J[6, 3] = 0.0
                J[6, 4] = gc + wsrf * gs + wveg[1] * gv[1]
                J[6, 5] = 0.0
                J[6, 6] = -Dveg[1] * wveg[1] * gv[1]
                f[0] = -(Rsrf - Gsrf - Hsrf - Lsrf * Esrf)
                f[1] = -(
                    Rveg[0] - Hveg[0] - Lcan[0] * Eveg[0]
                    - cveg[0] * (Tveg[0] - Tveg0[0]) / dt
                )
                f[2] = -(
                    Rveg[1] - Hveg[1] - Lcan[1] * Eveg[1]
                    - cveg[1] * (Tveg[1] - Tveg0[1]) / dt
                )
                f[3] = -(H - Hcan - Hveg[0]) / (rho * CP)
                f[4] = -(Hcan - Hsrf - Hveg[1]) / (rho * CP)
                f[5] = -(E - Ecan - Eveg[0]) / rho
                f[6] = -(Ecan - Esrf - Eveg[1]) / rho
                x = ludcmp(J, f)
                dTs = x[0]
                dQc[0] = x[1]
                dTc[0] = x[2]
                dTv[0] = x[3]
                dQc[1] = x[4]
                dTc[1] = x[5]
                dTv[1] = x[6]
                dEs = rho * wsrf * gs * (Dsrf * dTs - dQc[1])
                dEv[0] = rho * wveg[0] * gv[0] * (Dveg[0] * dTv[0] - dQc[0])
                dEv[1] = rho * wveg[1] * gv[1] * (Dveg[1] * dTv[1] - dQc[1])
                dGs = 2.0 * ks1 * dTs / Ds1
                dHs = rho * CP * gs * (dTs - dTc[1])
                dHv[0] = rho * CP * gv[0] * (dTv[0] - dTc[0])
                dHv[1] = rho * CP * gv[1] * (dTv[1] - dTc[1])
    
                # Surface melting
                if Tsrf + dTs > TM and Sice[0] > 0.0:
                    Melt = array_sum(Sice) / dt
                    f[0] = f[0] + LF * Melt
                    x = ludcmp(J, f)
                    dTs = x[0]
                    dQc[0] = x[1]
                    dTc[0] = x[2]
                    dTv[0] = x[3]
                    dQc[1] = x[4]
                    dTc[1] = x[5]
                    dTv[1] = x[6]
                    dEs = rho * wsrf * gs * (Dsrf * dTs - dQc[1])
                    dEv[0] = rho * wveg[0] * gv[0] * (
                        Dveg[0] * dTv[0] - dQc[0]
                    )
                    dEv[1] = rho * wveg[1] * gv[1] * (
                        Dveg[1] * dTv[1] - dQc[1]
                    )
                    dGs = 2.0 * ks1 * dTs / Ds1
                    dHs = rho * CP * gs * (dTs - dTc[1])
                    dHv[0] = rho * CP * gv[0] * (dTv[0] - dTc[0])
                    dHv[1] = rho * CP * gv[1] * (dTv[1] - dTc[1])
                    if Tsrf + dTs < TM:
                        Qsrf = qsat(Ps, TM)
                        Esrf = rho * wsrf * gs * (Qsrf - Qcan[1])
                        Hsrf = rho * CP * gs * (TM - Tcan[1])
                        Rsrf = Rsrf + SB * fourth(Tsrf) - SB * fourth(TM)
                        Rveg[0] = (
                            Rveg[0] + (1.0 - tdif[0]) * tdif[1] * SB
                            * (fourth(TM) - fourth(Tsrf))
                        )
                        Rveg[1] = (
                            Rveg[1] + (1.0 - tdif[1]) * SB
                            * (fourth(TM) - fourth(Tsrf))
                        )
                        J[0, 0] = -1.0
                        J[1, 0] = 0.0
                        J[2, 0] = 0.0
                        J[3, 0] = 0.0
                        J[4, 0] = 0.0
                        J[5, 0] = 0.0
                        J[6, 0] = 0.0
                        f[0] = -(Rsrf - Gsrf - Hsrf - Lsrf * Esrf)
                        f[1] = -(
                            Rveg[0] - Hveg[0] - Lcan[0] * Eveg[0]
                            - cveg[0] * (Tveg[0] - Tveg0[0]) / dt
                        )
                        f[2] = -(
                            Rveg[1] - Hveg[1] - Lcan[1] * Eveg[1]
                            - cveg[1] * (Tveg[1] - Tveg0[1]) / dt
                        )
                        f[3] = -(H - Hcan - Hveg[0]) / (rho * CP)
                        f[4] = -(Hcan - Hsrf - Hveg[1]) / (rho * CP)
                        f[5] = -(E - Ecan - Eveg[0]) / rho
                        f[6] = -(Ecan - Esrf - Eveg[1]) / rho
                        x = ludcmp(J, f)
                        Melt = x[0] / LF
                        dQc[0] = x[1]
                        dTc[0] = x[2]
                        dTv[0] = x[3]
                        dQc[1] = x[4]
                        dTc[1] = x[5]
                        dTv[1] = x[6]
                        dTs = TM - Tsrf
                        dEs = 0.0
                        dEv[0] = rho * wveg[0] * gv[0] * (
                            Dveg[0] * dTv[0] - dQc[0]
                        )
                        dEv[1] = rho * wveg[1] * gv[1] * (
                            Dveg[1] * dTv[1] - dQc[1]
                        )
                        dGs = 0.0
                        dHs = 0.0
                        dHv[0] = rho * CP * gv[0] * (dTv[0] - dTc[0])
                        dHv[1] = rho * CP * gv[1] * (dTv[1] - dTc[1])
    
                LWout = (
                    (1.0 - tdif[0]) * SB * fourth(Tveg[0])
                    + (1.0 - tdif[1]) * tdif[0] * SB * fourth(Tveg[0])
                    + tdif[0] * tdif[1] * SB * fourth(Tsrf)
                )
                LWsub = (
                    tdif[0] * tdif[1] * LW
                    + (1.0 - tdif[0]) * tdif[1] * SB * fourth(Tveg[0])
                    + (1.0 - tdif[1]) * SB * fourth(Tveg[1])
                )
    
            # Update vegetation temperatures and fluxes
            for n in range(ncnpy):
                Eveg[n] = Eveg[n] + dEv[n]
                Hveg[n] = Hveg[n] + dHv[n]
                Qcan[n] = Qcan[n] + dQc[n]
                Tcan[n] = Tcan[n] + dTc[n]
                Tveg[n] = Tveg[n] + dTv[n]
    
            # Update surface temperature and fluxes
            Esrf = Esrf + dEs
            Gsrf = Gsrf + dGs
            Hsrf = Hsrf + dHs
            Tsrf = Tsrf + dTs
            ebal = (
                SWsrf + LWsub - SB * fourth(Tsrf) - Gsrf - Hsrf
                - Lsrf * Esrf - LF * Melt
            )
            if ne > 4 and abs(ebal) < 0.01:
                break

    # Sublimation limited by available snow.  Ssub can be a round-off
    # residual when all the snow melts; treat values below this tolerance as
    # zero so that fastmath's reassociation cannot flip the branch and zero
    # the latent heat flux.  The tolerance is negligible in mass units
    # (1e-12 kg/m^2).
    subl = 0.0
    snow_available = array_sum(Sice)
    Ssub = snow_available - Melt * dt
    if Ssub > 1e-12 or Tsrf < TM:
        Esrf = min(Esrf, Ssub / dt)
        subl = Esrf
    if VAI > 0.0:
        for n in range(ncnpy):
            if Sveg[n] > 1e-12 or Tveg[n] < TM:
                Eveg[n] = min(Eveg[n], Sveg[n] / dt)
                subl = subl + Eveg[n]


    # Fluxes to the atmosphere
    E = Esrf + array_sum(Eveg)
    H = Hsrf + array_sum(Hveg)
    LE = Lsrf * Esrf
    for n in range(ncnpy):
        LE = LE + Lcan[n] * Eveg[n]

    # Diagnostics
    if VAI == 0.0:
        LWout = SB * fourth(Tsrf)
        ustar = VKMAN * Ua / (
            math.log(zU1 / z0g) - psim(zU1, rL) + psim(z0g, rL)
        )
        Usub = (ustar / VKMAN) * (
            math.log(params.zsub / z0g)
            - psim(params.zsub, rL) + psim(z0g, rL)
        )
        gs = VKMAN * ustar / (
            math.log(params.zsub / z0h)
            - psih(params.zsub, rL) + psih(z0h, rL)
        )
    else:
        Uc = math.exp(params.wcan * (params.hbas / vegh - 1.0)) * Uh
        Usub = (
            fveg * Uc * math.log(params.zsub / z0g)
            / math.log(params.hbas / z0g)
            + (1.0 - fveg) * Ua
            * (
                math.log(params.zsub / z0g)
                - psim(params.zsub, rL) + psim(z0g, rL)
            )
            / (
                math.log(zU1 / z0g)
                - psim(zU1, rL) + psim(z0g, rL)
            )
        )
        rd = (
            math.log(params.zsub / z0g)
            * math.log(params.zsub / z0h)
            / (VKMAN ** 2 * Uc)
        )
        ro = (
            math.log(params.zsub / z0h)
            - psih(params.zsub, rL) + psih(z0h, rL)
        ) / (VKMAN * ustar)
        gs = fveg / rd + (1.0 - fveg) / ro
    Tsub = Tsrf - Hsrf / (CP * rho * gs)

    return (Tsrf, Esrf, Eveg, Gsrf, H, LE, LWout, LWsub, Melt, subl,
            Tsub, Usub)

