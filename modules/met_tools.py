#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of functions to interact with the meteorological data and calculate
other parameters related with the snowpack

Author: Esteban Alonso González - alonsoe@cesbio.cnes.fr
"""
import numpy as np
from scipy.optimize import newton
from scipy.special import expit, logit
import constants as cnt
import config as cfg
import modules.filters as flt
import modules.spatialMuSA as spM


def pp_psychrometric(ta2, rh2, precc):

    ta2 = np.asarray(ta2)
    rh2 = np.asarray(rh2)
    precc = np.asarray(precc)

    ta2_c = ta2 - cnt.KELVING_CONVER             # temp in [ºC]

    # L = latent heat of sublimation or vaporisation [J kg**-1]
    latn_h = np.where(ta2_c >= 0,
                      1000 * (2501 - (2.361 * ta2_c)),
                      ta2_c)
    latn_h = np.where(latn_h < 0,
                      1000 * (2834.1 - 0.29 * latn_h - 0.004 * latn_h**2),
                      latn_h)

    # thermal conduc of air [J (msK)**-1]
    thr_cond = 0.000063 * ta2 + 0.00673
    # diffusivity of vapour [m**2 s**-1]
    dif_wv = 2.06E-5 * (ta2/273.15)**1.75
    # vapour pressure [KPa]
    e_press = (rh2/100) * 0.611 * np.exp((17.3 * ta2_c)/(ta2_c + 237.3))
    # vapour pressure [Pa]
    e_press = e_press * 1000
    # water vapour density in atmosphere kg m**-3
    pta = (cnt.MW * e_press) / (cnt.R * ta2)

    def eti(surf_ti):  # Vapour pressure ecuation
        return 0.611 * 1000 * np.exp((17.3 * (surf_ti - 273) /
                                      ((surf_ti - 273) + 237.3)))

    def pti(surf_ti):  # Vapour density ecuation
        return (cnt.MW * eti(surf_ti)) / (cnt.R * surf_ti)

    def ti_ecuation(surf_ti):  # Objetive function to solve
        return ta2 + (dif_wv / thr_cond) * latn_h * (pta -
                                                     pti(surf_ti)) - surf_ti

    # Hidrometeor Surface temperature [ºK]
    surf_ti = newton(ti_ecuation, ta2)
    surf_ti = np.where(np.isfinite(surf_ti), surf_ti, ta2)  # if root fails
    # Liquid precipitation fraction
    rain_fr = 1 / (1 + 2.50286 * 0.125006**(surf_ti - cnt.KELVING_CONVER))

    liquid_prec = precc * rain_fr
    solid_prec = precc * (1-rain_fr)
    return liquid_prec, solid_prec


def pp_temp_thld_log(ta2, precc):

    ta2 = np.asarray(ta2)
    precc = np.asarray(precc)

    m_prec = 0.3051  # Parameter influencing the temp range with mixed precc
    thres_prec = 273.689  # Threshold where 50% falls as rain/snow
    # Parameter influencing the temperature range with mixed precipitation

    Tp = (ta2 - thres_prec)/m_prec
    p_multi = np.exp(Tp)/(1 + np.exp(Tp))

    liquid_prec = p_multi * precc
    solid_prec = precc - liquid_prec

    solid_prec[solid_prec < 0] = 0

    return liquid_prec, solid_prec


def linear_liston(ta2, precc):

    ta2 = np.asarray(ta2)
    precc = np.asarray(precc)

    temp_C = ta2-cnt.KELVING_CONVER

    Tair_C_center = 1.1
    slope = -0.3
    b = 0.5 - slope*Tair_C_center
    snowfall_frac = np.array(slope * temp_C + b)
    snowfall_frac[snowfall_frac > 1] = 1.
    snowfall_frac[snowfall_frac < 0] = 0.

    solid_prec = precc * snowfall_frac
    liquid_prec = precc * (1 - snowfall_frac)
    return liquid_prec, solid_prec


def glogit(x, bPmin, bPmax):
    """ Generalized logit that transforms a logit-normal to a normal"""

    # HACK: make it numerically stable
    if (x == bPmin).any():
        x[x == bPmin] = x[x == bPmin] + np.abs(np.spacing(x[x == bPmin])*100)
    if (x == bPmax).any():
        x[x == bPmax] = x[x == bPmax] - np.abs(np.spacing(x[x == bPmax])*100)

    xs = (x-bPmin)/(bPmax-bPmin)  # Scale x\in(a,b) to be between (0,1) instead
    xt = logit(xs)
    return xt


def gexpit(xt, bPmin, bPmax):
    """ Generalized expit that transforms a normal to a logit-normal"""
    # HACK: make it numerically stable
    if (xt == bPmin).any():
        xt[xt == bPmin] = xt[xt == bPmin] +\
            np.abs(np.spacing(xt[xt == bPmin])*100)
    if (xt == bPmax).any():
        xt[xt == bPmax] = xt[xt == bPmax] -\
            np.abs(np.spacing(xt[xt == bPmax])*100)

    xs = expit(xt)
    x = (bPmax-bPmin)*xs+bPmin
    return x


def create_noise(perturbation_strategy, n_steps, mean, std_dev, bPmin, bPmax):

    if(cfg.seed is not None):
        np.random.seed(cfg.seed)

    if perturbation_strategy == "normal":
        noise = np.random.normal(mean, std_dev, 1)
        noise = np.repeat(noise, n_steps)

    elif perturbation_strategy == "lognormal":
        noise = np.random.lognormal(mean, std_dev, 1)
        noise = np.repeat(noise, n_steps)

    elif perturbation_strategy in ["logitnormal_mult",
                                   "logitnormal_adi"]:

        norm_noise = np.random.normal(mean, std_dev, 1)
        norm_noise = np.repeat(norm_noise, n_steps)
        noise = gexpit(norm_noise, bPmin, bPmax)

    else:
        raise Exception("choose the shape of the perturbation parameters")
    return noise


def add_process_noise(noise_coef, var, strategy):

    dyn_noise = cnt.dyn_noise
    upper_bounds = cnt.upper_bounds
    lower_bounds = cnt.lower_bounds

    # Not to add process noise in spatial version yet
    if cfg.implementation == 'Spatial_propagation':
        return noise_coef

    if cfg.add_dynamic_noise:

        # Add process noise
        tmp_proc_noise = dyn_noise[var]
        add_vals = np.random.normal(0, tmp_proc_noise, len(noise_coef))

        if strategy == 'normal':

            noise_coef = noise_coef + np.cumsum(add_vals)

        elif strategy == 'lognormal':

            noise_coef = np.log(noise_coef) + np.cumsum(add_vals)
            noise_coef = np.exp(noise_coef)

        elif strategy in ["logitnormal_mult",
                          "logitnormal_adi"]:
            noise_coef = glogit(noise_coef,
                                lower_bounds[var],
                                upper_bounds[var]) + np.cumsum(add_vals)
            noise_coef = gexpit(noise_coef,
                                lower_bounds[var],
                                upper_bounds[var])
        else:
            raise Exception("Bad perturbation strategy")

    else:
        pass

    return noise_coef


def perturb_parameters(main_forcing, lat_idx=None, lon_idx=None, member=None,
                       noise=None, update=False, readGSC=False,
                       GSC_filename=None):

    forcing_copy = main_forcing.copy()

    vars_to_perturbate = cfg.vars_to_perturbate
    perturbation_strategy = cfg.perturbation_strategy
    mean_errors = cnt.mean_errors
    sd_errors = cnt.sd_errors
    upper_bounds = cnt.upper_bounds
    lower_bounds = cnt.lower_bounds

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
            parameter = spM.read_parameter(GSC_filename, lat_idx, lon_idx,
                                           var_tmp, member)
            noise_coef = np.repeat(parameter, n_steps)

        elif update:
            noise_coef = noise[idv]
            noise_coef = np.repeat(noise_coef, n_steps)
        else:

            noise_coef = create_noise(strategy_tmp, n_steps,
                                      mean_errors[var_tmp],
                                      sd_errors[var_tmp],
                                      lower_bounds[var_tmp],
                                      upper_bounds[var_tmp])

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

        var_temp = [noise_dict[mbr][var].copy() for
                    mbr in range(ensemble_members)]
        var_temp = np.asarray(var_temp)
        var_temp = np.squeeze(var_temp)

        # Transform this from log-space to make it normally distributed
        if perturbation_strategy[count] == "lognormal":
            var_temp = np.log(var_temp)

        if perturbation_strategy[count] == ["logitnormal_mult",
                                            "logitnormal_adi"]:
            var_temp = glogit(var_temp,
                              upper_bounds[var],
                              lower_bounds[var])

        # reduce the timeserie to its mean
        mu = np.mean(np.average(var_temp, axis=0, weights=wgth))
        sigma = np.mean(flt.weighted_std(var_temp, axis=0, weights=wgth))

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

        new_pert = mu+sigma*np.random.randn(1)
        if var == "lognormal":
            # re-draw in log-space
            new_pert = np.exp(new_pert)

        if perturbation_strategy[count] == ["logitnormal_mult",
                                            "logitnormal_adi"]:

            varindict = cfg.vars_to_perturbate[count]
            new_pert = gexpit(new_pert,
                              upper_bounds[varindict],
                              lower_bounds[varindict])

        storage[count] = new_pert

    return storage
