#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions to interact with the meteorological data and calculate
other parameters related with the snowpack

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import numpy as np
from scipy.special import expit, logit
import constants as cnt
import config as cfg
from statsmodels.stats.weightstats import DescrStatsW
import modules.spatialMuSA as spM


def pp_psychrometric(ta2, rh2, precc, ret_fr=False):

    ta2 = np.asarray(ta2, dtype=float)
    rh2 = np.asarray(rh2, dtype=float)
    precc = np.asarray(precc, dtype=float)

    # Work on flat, broadcast-compatible arrays: index assignment is used
    # below and 0-d arrays do not support it. The original shape is restored
    # before returning.
    shape = np.broadcast_shapes(ta2.shape, rh2.shape, precc.shape)
    ta2 = np.broadcast_to(ta2, shape).reshape(-1)
    rh2 = np.broadcast_to(rh2, shape).reshape(-1)
    precc = np.broadcast_to(precc, shape).reshape(-1)

    tc = ta2 - cnt.KELVING_CONVER

    # Latent heat [J kg-1]
    latn_h = np.where(
        tc >= 0,
        1000.0 * (2501.0 - 2.361 * tc),
        1000.0 * (2834.1 - 0.29 * tc - 0.004 * tc**2),
    )

    # Thermal conductivity of air [W m-1 K-1]
    thr_cond = 0.000063 * ta2 + 0.00673

    # Water vapour diffusivity [m2 s-1]
    dif_wv = 2.06e-5 * (ta2 / 273.15) ** 1.75

    # Vapour pressure [Pa]
    e_press = (rh2 / 100.0) * 611.0 * np.exp((17.3 * tc) / (tc + 237.3))

    # Vapour density [kg m-3]
    pta = (cnt.MW * e_press) / (cnt.R * ta2)

    # Surface temperature: solve
    #     f(T) = Ta + (Dv / k) * L * (pta - pti(T)) - T = 0
    # f is strictly decreasing in T. Instead of one scalar brentq per
    # timestep (millions of Python callbacks), use a vectorised bisection.
    #
    # Outside [T_SNOW, T_RAIN] the rain fraction is 0 or 1 to within ~1e-9,
    # which is far below any meaningful precipitation tolerance, so those
    # points are resolved without iterating. For the rest, the sign of f at
    # both bounds also gives a tight initial bracket.
    T_SNOW = 263.15  # rain_fr < 4e-10 below this surface temperature
    T_RAIN = 283.15  # 1 - rain_fr < 3e-9 above this surface temperature
    T_LO, T_HI = 223.15, 323.15

    alpha = (dif_wv / thr_cond) * latn_h

    def f_surface(T):
        # pti(T) inlined (formerly eti() + pti()) to avoid two extra
        # Python calls per evaluation.
        t_c = T - cnt.KELVING_CONVER
        pti = (cnt.MW * 611.0 * np.exp((17.3 * t_c) / (t_c + 237.3))) / (
            cnt.R * T
        )
        return ta2 + alpha * (pta - pti) - T

    f_lo = f_surface(T_LO)
    f_hi = f_surface(T_HI)
    # Same domain check as brentq: a root exists only when f changes sign.
    bracket = (f_lo >= 0.0) & (f_hi <= 0.0)

    f_snow = f_surface(T_SNOW)
    f_rain = f_surface(T_RAIN)

    surf_ti = ta2.copy()  # fallback, as in the previous ValueError branch
    all_snow = bracket & (f_snow <= 0.0)
    all_rain = bracket & (f_rain >= 0.0)
    mixed = bracket & ~all_snow & ~all_rain

    surf_ti[all_snow] = T_SNOW
    surf_ti[all_rain] = T_RAIN

    if mixed.any():
        ta_m = ta2[mixed]
        pta_m = pta[mixed]
        alpha_m = alpha[mixed]

        lo = np.full(ta_m.shape, T_SNOW)
        hi = np.full(ta_m.shape, T_RAIN)

        # 20 K / 2**48 ~ 7e-14 K: well below the brentq tolerance.
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            mid_c = mid - cnt.KELVING_CONVER
            pti_mid = (
                cnt.MW * 611.0 * np.exp((17.3 * mid_c) / (mid_c + 237.3))
            ) / (cnt.R * mid)
            below = (ta_m + alpha_m * (pta_m - pti_mid) - mid) < 0.0
            hi = np.where(below, mid, hi)
            lo = np.where(below, lo, mid)

        surf_ti[mixed] = 0.5 * (lo + hi)

    # Liquid fraction
    rain_fr = 1.0 / (
        1.0
        + 2.50286
        * np.power(
            0.125006,
            surf_ti - cnt.KELVING_CONVER,
        )
    )

    rain_fr = np.clip(rain_fr, 0.0, 1.0)

    liquid_prec = precc * rain_fr
    solid_prec = precc * (1.0 - rain_fr)

    thresh = 0.1 / 3600.0

    liquid_prec = np.where(liquid_prec < thresh, 0.0, liquid_prec)
    solid_prec = np.where(solid_prec < thresh, 0.0, solid_prec)

    liquid_prec = liquid_prec.reshape(shape)
    solid_prec = solid_prec.reshape(shape)
    rain_fr = rain_fr.reshape(shape)

    if ret_fr:
        return liquid_prec, solid_prec, rain_fr

    return liquid_prec, solid_prec


def pp_temp_thld_log(ta2, precc):

    m_prec = 0.3051  # Parameter influencing the temp range with mixed precc
    thres_prec = 273.689  # Threshold where 50% falls as rain/snow
    # Parameter influencing the temperature range with mixed precipitation

    Tp = (ta2 - thres_prec) / m_prec
    p_multi = np.exp(Tp) / (1 + np.exp(Tp))

    liquid_prec = p_multi * precc
    solid_prec = precc - liquid_prec

    solid_prec[solid_prec < 0] = 0

    return liquid_prec, solid_prec


def pres_from_dem(topo):
    # From Micromet (Liston & Elder, 2006 https://doi.org/10.1175/JHM486.1)

    one_atmos = 101300.0
    scale_ht = 8000.0
    sfc_pressure = one_atmos * np.exp(-topo / scale_ht)
    return sfc_pressure


def linear_liston(ta2, precc):

    temp_C = ta2 - cnt.KELVING_CONVER

    Tair_C_center = 1.1
    slope = -0.3
    b = 0.5 - slope * Tair_C_center
    snowfall_frac = np.array(slope * temp_C + b)
    snowfall_frac[snowfall_frac > 1] = 1.0
    snowfall_frac[snowfall_frac < 0] = 0.0

    solid_prec = precc * snowfall_frac
    liquid_prec = precc * (1 - snowfall_frac)
    return liquid_prec, solid_prec


def glogit(x, bPmin, bPmax):
    """Generalized logit that transforms a logit-normal to a normal"""

    # HACK: make it numerically stable
    if (x == bPmin).any():
        x[x == bPmin] = x[x == bPmin] + np.abs(np.spacing(x[x == bPmin]) * 100)
    if (x == bPmax).any():
        x[x == bPmax] = x[x == bPmax] - np.abs(np.spacing(x[x == bPmax]) * 100)

    xs = (x - bPmin) / (
        bPmax - bPmin
    )  # Scale x\in(a,b) to be between (0,1) instead
    xt = logit(xs)
    return xt


def gexpit(xt, bPmin, bPmax):
    """Generalized expit that transforms a normal to a logit-normal"""
    # HACK: make it numerically stable
    if (xt == bPmin).any():
        xt[xt == bPmin] = xt[xt == bPmin] + np.abs(
            np.spacing(xt[xt == bPmin]) * 100
        )
    if (xt == bPmax).any():
        xt[xt == bPmax] = xt[xt == bPmax] - np.abs(
            np.spacing(xt[xt == bPmax]) * 100
        )

    xs = expit(xt)
    x = (bPmax - bPmin) * xs + bPmin
    return x


def create_noise(perturbation_strategy, n_steps, mean, std_dev, var):

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    if perturbation_strategy == "normal":
        noise = np.random.normal(mean, std_dev, 1)
        noise = np.repeat(noise, n_steps)

    elif perturbation_strategy == "lognormal":
        noise = np.random.lognormal(mean, std_dev, 1)
        noise = np.repeat(noise, n_steps)

    elif perturbation_strategy in ["logitnormal_mult", "logitnormal_adi"]:

        bPmax = cnt.upper_bounds[var]
        bPmin = cnt.lower_bounds[var]

        norm_noise = np.random.normal(mean, std_dev, 1)
        norm_noise = np.repeat(norm_noise, n_steps)
        noise = gexpit(norm_noise, bPmin, bPmax)

    else:
        raise Exception("choose the shape of the perturbation parameters")
    return noise


def create_noise_from_posterior(posteriors, perturbation_strategy, var_tmp):

    if cfg.seed is not None:
        np.random.seed(cfg.seed)

    # Retrieve posterior timeseries
    post_mean = posteriors[var_tmp + "_noise_mean"].to_numpy()
    post_std_dev = posteriors[var_tmp + "_noise_sd"].to_numpy()

    # take unique values (assim steps)
    post_mean_vals = np.unique(post_mean)
    # NOTE: the standard deviations can be 0, since they can be collapsed
    # ensembles. Thus npunique is not safe. The trick is done through the
    # averages, which are very unlikely (impossible in practice) to
    # be equal between DAWs.
    # post_post_std_dev = np.unique(post_std_dev)

    if len(posteriors) == len(post_mean_vals):  # dynamic noise case
        post_mean_vals = np.mean(post_mean_vals)
        # number of parameters to repeat = len of forcing
        counts = len(posteriors)

    else:  # count the nymber of steps in each DAW
        counts = np.array(
            [np.sum(post_mean == value) for value in post_mean_vals]
        )

    # This part is necesary in case of total collapse of one or several DAWs
    idsDAWS = np.repeat(range(len(counts)), counts)
    unique_indices = np.unique(idsDAWS)
    post_post_std_dev = np.zeros(unique_indices.shape)

    # Calculate the means for each unique index
    for i, index in enumerate(unique_indices):
        # combined standar deviation
        k = np.sum(idsDAWS == index)
        post_post_std_dev[i] = np.sqrt(
            np.sum(
                post_std_dev[idsDAWS == index] * post_std_dev[idsDAWS == index]
            )
            / k
        )

    complet_noise = []

    for n in range(post_mean_vals.size):  # loop over DAW

        mean, std_dev = post_mean_vals[n], post_post_std_dev[n]
        n_steps = counts[0]

        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        if perturbation_strategy == "normal":
            noise = np.random.normal(mean, std_dev, 1)
            noise = np.repeat(noise, n_steps)

        elif perturbation_strategy == "lognormal":
            noise = np.random.lognormal(mean, std_dev, 1)
            noise = np.repeat(noise, n_steps)

        elif perturbation_strategy in ["logitnormal_mult", "logitnormal_adi"]:

            bPmax = cnt.upper_bounds[var_tmp]
            bPmin = cnt.lower_bounds[var_tmp]

            norm_noise = np.random.normal(mean, std_dev, 1)
            norm_noise = np.repeat(norm_noise, n_steps)
            noise = gexpit(norm_noise, bPmin, bPmax)

        else:
            raise Exception("choose the shape of the perturbation parameters")

        complet_noise.append(noise)

    return np.concatenate(complet_noise)


def add_process_noise(noise_coef, var, strategy):

    dyn_noise = cnt.dyn_noise
    upper_bounds = cnt.upper_bounds
    lower_bounds = cnt.lower_bounds

    # Not to add process noise in spatial version yet
    if cfg.implementation == "Spatial_propagation":
        return noise_coef

    if cfg.add_dynamic_noise:

        # Add process noise
        tmp_proc_noise = dyn_noise[var]
        add_vals = np.random.normal(0, tmp_proc_noise, len(noise_coef))

        if strategy == "normal":

            noise_coef = noise_coef + np.cumsum(add_vals)

        elif strategy == "lognormal":

            noise_coef = np.log(noise_coef) + np.cumsum(add_vals)
            noise_coef = np.exp(noise_coef)

        elif strategy in ["logitnormal_mult", "logitnormal_adi"]:
            noise_coef = glogit(
                noise_coef, lower_bounds[var], upper_bounds[var]
            ) + np.cumsum(add_vals)
            noise_coef = gexpit(
                noise_coef, lower_bounds[var], upper_bounds[var]
            )
        else:
            raise Exception("Bad perturbation strategy")

    else:
        pass

    return noise_coef


def perturb_parameters(
    main_forcing,
    lat_idx=None,
    lon_idx=None,
    posteriors=None,
    member=None,
    noise=None,
    update=False,
    readGSC=False,
    GSC_filename=None,
):

    forcing_copy = main_forcing.copy()

    vars_to_perturbate = cfg.vars_to_perturbate
    perturbation_strategy = cfg.perturbation_strategy
    mean_errors = cnt.mean_errors
    sd_errors = cnt.sd_errors
    n_steps = forcing_copy.shape[0]

    # Create dict to store the perturbing parameters
    noise_storage = dict.fromkeys(vars_to_perturbate, None)

    if len(vars_to_perturbate) != len(perturbation_strategy):
        raise Exception("""Length of vars_to_perturbate different to
                        perturbation_strategy""")
    if update and len(vars_to_perturbate) != len(noise):
        raise Exception("""Something wrong with kalman noise""")

    # Loop over perturbation vars
    for idv in range(len(vars_to_perturbate)):

        var_tmp = vars_to_perturbate[idv]
        strategy_tmp = perturbation_strategy[idv]

        if readGSC:
            parameter = spM.read_parameter(
                GSC_filename, lat_idx, lon_idx, var_tmp, member
            )
            noise_coef = np.repeat(parameter, n_steps)

        elif update:
            noise_coef = noise[idv]
            noise_coef = np.repeat(noise_coef, n_steps)
        else:
            if cfg.load_prev_run:
                noise_coef = create_noise_from_posterior(
                    posteriors, strategy_tmp, var_tmp
                )
            else:
                noise_coef = create_noise(
                    strategy_tmp,
                    n_steps,
                    mean_errors[var_tmp],
                    sd_errors[var_tmp],
                    var_tmp,
                )

        noise_coef = add_process_noise(noise_coef, var_tmp, strategy_tmp)
        # If lognormal perturbation multiplicate, else add
        if strategy_tmp in ["lognormal", "logitnormal_mult"]:
            forcing_copy[var_tmp] = forcing_copy[var_tmp].values * noise_coef
        else:
            forcing_copy[var_tmp] = forcing_copy[var_tmp].values + noise_coef

        # store the noise
        noise_storage[var_tmp] = noise_coef

    return forcing_copy, noise_storage


def get_shape_from_noise(noise_dict, wgth, lowNeff):

    vars_to_perturbate = cfg.vars_to_perturbate
    perturbation_strategy = cfg.perturbation_strategy
    ensemble_members = cfg.ensemble_members
    upper_bounds = cnt.upper_bounds
    lower_bounds = cnt.lower_bounds

    storage = np.ones((len(vars_to_perturbate), 2))

    for count, var in enumerate(vars_to_perturbate):

        var_temp = [
            noise_dict[mbr][var].copy() for mbr in range(ensemble_members)
        ]
        var_temp = np.asarray(var_temp)
        var_temp = np.squeeze(var_temp)

        # Transform this from log-space to make it normally distributed
        if perturbation_strategy[count] == "lognormal":
            var_temp = np.log(var_temp)

        if perturbation_strategy[count] == [
            "logitnormal_mult",
            "logitnormal_adi",
        ]:
            var_temp = glogit(var_temp, upper_bounds[var], lower_bounds[var])

        # reduce the timeserie to its mean

        d1 = DescrStatsW(np.squeeze(var_temp), weights=wgth)
        mu = np.mean(d1.mean)
        sigma = np.mean(d1.std)

        # Fix to recover from collapse through particle rejuvenation
        if lowNeff:
            sigma = cnt.sd_errors[var] * cnt.sdfrac

        storage[count, 0] = mu
        storage[count, 1] = sigma

    return storage


def redraw(func_shape):

    perturbation_strategy = cfg.perturbation_strategy
    upper_bounds = cnt.upper_bounds
    lower_bounds = cnt.lower_bounds

    storage = np.ones(len(perturbation_strategy))

    for count, var in enumerate(perturbation_strategy):

        mu = func_shape[count, 0].copy()
        sigma = func_shape[count, 1].copy()

        new_pert = mu + sigma * np.random.randn(1)
        if var == "lognormal":
            # re-draw in log-space
            new_pert = np.exp(new_pert)

        if perturbation_strategy[count] == [
            "logitnormal_mult",
            "logitnormal_adi",
        ]:

            varindict = cfg.vars_to_perturbate[count]
            new_pert = gexpit(
                new_pert, upper_bounds[varindict], lower_bounds[varindict]
            )

        storage[count] = new_pert

    return storage
