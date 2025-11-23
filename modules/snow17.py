#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The snow17 model (original version)

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""
import numpy as np
import numba as nb
import constants as cnt
import config as cfg


@nb.njit(fastmath=True, cache=True)
def snow17(time, prec, tair, p_atm, lat, init,
           uadj, mbase, mfmax, mfmin, tipm,
           nmf, plwhc, pxtemp, pxtemp1, pxtemp2):
    """
    Snow-17 accumulation and ablation model. This version of Snow-17 is
    intended for use at a point location.
    The time steps for precipitation and temperature must be equal for this
    code.

    Modified from original code available at https://github.com/UW-Hydro/tonic

    """

    dt = (cfg.dt/3600)
    rvs = 1

    # Convert to numpy array if scalars
    time = np.asarray(time)
    prec = np.asarray(prec)
    tair = np.asarray(tair)

    # Initialization
    # Antecedent Temperature Index, deg C
    ait = init[0]
    # Liquid water capacity
    w_qx = init[1]
    # Liquid water held by the snow (mm)
    w_q = init[2]
    # accumulated water equivalent of the iceportion of the snow cover (mm)
    w_i = init[3]
    # Heat deficit, also known as NEGHS, Negative Heat Storage
    deficit = init[4]

    # number of time steps
    nsteps = len(time)
    model_swe = np.zeros(nsteps)
    outflow = np.zeros(nsteps)

    # Stefan-Boltzman constant (mm/K/hr)
    stefan = 6.12 * (10 ** (-10))

    transitionx = [pxtemp1, pxtemp2]
    transitiony = [1.0, 0.0]

    tipm_dt = 1.0 - ((1.0 - tipm) ** (dt / 6))

    # Model Execution
    for i, jday in enumerate(time):
        mf = melt_function(jday, dt, lat[i], mfmax, mfmin)

        # air temperature at this time step (deg C)
        t_air_mean = tair[i]
        # precipitation at this time step (mm)
        precip = prec[i]

        # Divide rain and snow
        if rvs == 0:
            if t_air_mean <= pxtemp:
                # then the air temperature is cold enough for snow to occur
                fracsnow = 1.0
            else:
                # then the air temperature is warm enough for rain
                fracsnow = 0.0
        elif rvs == 1:
            if t_air_mean <= pxtemp1:
                fracsnow = 1.0
            elif t_air_mean >= pxtemp2:
                fracsnow = 0.0
            else:
                fracsnow = np.interp(t_air_mean, transitionx, transitiony)
        elif rvs == 2:
            fracsnow = 1.0
        else:
            raise ValueError('Invalid rain vs snow option')

        fracrain = 1.0 - fracsnow

        # Snow Accumulation
        # water equivalent of new snowfall (mm)
        pn = precip * fracsnow
        # w_i = accumulated water equivalent of the ice portion of the snow
        # cover (mm)
        w_i += pn
        e = 0.0
        # amount of precip (mm) that is rain during this time step
        rain = fracrain * precip

        # Temperature and Heat deficit from new Snow
        if t_air_mean < 0.0:
            t_snow_new = t_air_mean
            # delta_hd_snow = change in the heat deficit due to snowfall (mm)
            delta_hd_snow = - (t_snow_new * pn) / (80 / 0.5)
            t_rain = pxtemp
        else:
            t_snow_new = 0.0
            delta_hd_snow = 0.0
            t_rain = t_air_mean

        # Antecedent temperature Index
        if pn > (1.5 * dt):
            ait = t_snow_new
        else:
            # Antecedent temperature index
            ait = ait + tipm_dt * (t_air_mean - ait)
        if ait > 0:
            ait = 0

        # Heat Exchange when no Surface melt
        # delta_hd_t = change in heat deficit due to a temperature gradient(mm)
        delta_hd_t = nmf * (dt / 6.0) * ((mf) / mfmax) * (ait - t_snow_new)

        # Rain-on-snow melt
        # saturated vapor pressure at t_air_mean (mb)
        e_sat = 2.7489 * (10 ** 8) * np.exp(
            (-4278.63 / (t_air_mean + 242.792)))
        # 1.5 mm/ 6 hrs
        if rain > (0.25 * dt):
            # melt (mm) during rain-on-snow periods is:
            m_ros1 = np.maximum(
                stefan * dt * (((t_air_mean + 273) ** 4) - (273 ** 4)), 0.0)
            m_ros2 = np.maximum((0.0125 * rain * t_rain), 0.0)
            m_ros3 = np.maximum((8.5 * uadj *
                                (dt / 6.0) *
                                (((0.9 * e_sat) - 6.11) +
                                 (0.00057 * p_atm[i] * t_air_mean))),
                                0.0)
            m_ros = m_ros1 + m_ros2 + m_ros3
        else:
            m_ros = 0.0

        # Non-Rain melt
        if rain <= (0.25 * dt) and (t_air_mean > mbase):
            # melt during non-rain periods is:
            m_nr = (mf * (t_air_mean - mbase)) + (0.0125 * rain * t_rain)
        else:
            m_nr = 0.0

        # Ripeness of the snow cover
        melt = m_ros + m_nr
        if melt <= 0:
            melt = 0.0

        if melt < w_i:
            w_i = w_i - melt
        else:
            melt = w_i + w_q
            w_i = 0.0

        # qw = liquid water available melted/rained at the snow surface (mm)
        qw = melt + rain
        # w_qx = liquid water capacity (mm)
        w_qx = plwhc * w_i
        # deficit = heat deficit (mm)
        deficit += delta_hd_snow + delta_hd_t

        # limits of heat deficit
        if deficit < 0:
            deficit = 0.0
        elif deficit > 0.33 * w_i:
            deficit = 0.33 * w_i

        # Snow cover is ripe when both (deficit=0) & (w_q = w_qx)
        if w_i > 0.0:
            if (qw + w_q) > ((deficit * (1 + plwhc)) + w_qx):
                # THEN the snow is RIPE
                # Excess liquid water (mm)
                e = qw + w_q - w_qx - (deficit * (1 + plwhc))
                # fills liquid water capacity
                w_q = w_qx
                # w_i increases because water refreezes as heat deficit is
                # decreased
                w_i = w_i + deficit
                deficit = 0.0
            elif ((qw >= deficit) and
                  # ait((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx))):
                  ((qw + w_q) <= ((deficit * (1 + plwhc)) + w_qx))):
                # THEN the snow is NOT yet ripe, but ice is being melted
                e = 0.0
                w_q = w_q + qw - deficit
                # w_i increases because water refreezes as heat deficit is
                # decreased
                w_i = w_i + deficit
                deficit = 0.0
            else:
                # (qw < deficit) %elseif ((qw + w_q) < deficit):
                # THEN the snow is NOT yet ripe
                e = 0.0
                # w_i increases because water refreezes as heat deficit is
                # decreased
                w_i = w_i + qw
                deficit = deficit - qw
            swe = w_i + w_q
        else:
            e = qw
            swe = 0

        if deficit == 0:
            ait = 0

        # End of model execution
        model_swe[i] = swe  # total swe (mm) at this time step
        outflow[i] = e

        # create restart point
        init = np.array([ait, w_qx, w_q, w_i, deficit])

    return model_swe, model_swe/cnt.FIX_density/1000, outflow, init


@nb.njit(fastmath=True, cache=True)
def melt_function(jday, dt, lat, mfmax, mfmin):
    """
    Seasonal variation calcs - indexed for Non-Rain melt
    Parameters
    ----------
    jday : scalar
        julian day.
    dt : float
        Timestep in hours.
    lat : float
        Latitude of simulation point or grid cell.
    mfmax : float
        Maximum melt factor during non-rain periods (mm/deg C 6 hr) - in
        western facing slope assumed to occur on June 21. Default value of 1.05
        is from the American River Basin fromShamir & Georgakakos 2007.
    mfmin : float
        Minimum melt factor during non-rain periods (mm/deg C 6 hr) - in
        western facing slope assumed to occur on December 21. Default value of
        0.60 is from the American River Basin from Shamir & Georgakakos 2007.
    Returns
    ----------
    meltf : float
        Melt function for current timestep.
    """

    n_mar21 = jday - 80
    days = 365

    # seasonal variation
    sv = (0.5 * np.sin((n_mar21 * 2 * np.pi) / days)) + 0.5
    if lat < 54:
        # latitude parameter, av=1.0 when lat < 54 deg N
        av = 1.0
    else:
        if jday <= 77 or jday >= 267:
            # av = 0.0 from September 24 to March 18,
            av = 0.0
        elif jday >= 117 and jday <= 227:
            # av = 1.0 from April 27 to August 15
            av = 1.0
        elif jday >= 78 and jday <= 116:
            # av varies linearly between 0.0 and 1.0 from 3/19-4/26 and
            # between 1.0 and 0.0 from 8/16-9/23.
            av = np.interp(jday, [78, 116], [0, 1])
        elif jday >= 228 and jday <= 266:
            av = np.interp(jday, [228, 266], [1, 0])
    meltf = (dt / 6) * ((sv * av * (mfmax - mfmin)) + mfmin)

    return meltf
