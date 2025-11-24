#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main function performing DA over a cell

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""
import os
import warnings
import numpy as np
import pandas as pd
import config as cfg
import modules.internal_fns as ifn
import modules.filters as flt
from modules.internal_class import SnowEnsemble
from threadpoolctl import threadpool_limits

if cfg.numerical_model == 'FSM2':
    import modules.fsm_tools as model
elif cfg.numerical_model == 'dIm':
    import modules.dIm_tools as model
elif cfg.numerical_model == 'snow17':
    import modules.snow17_tools as model
else:
    raise Exception('Model not implemented')


def mean_daily(var_df_hourly):
    # Resample variables to be output to daily time step
    var_df_hourly['Date'] = pd.to_datetime(
        var_df_hourly['Date'], format='%d/%m/%Y-%H:%M')
    var_df_hourly.set_index('Date', inplace=True)

    # Resample to daily frequency and take the mean
    var_df_daily = var_df_hourly.resample('D').mean()

    var_df_hourly.reset_index(inplace=True)
    var_df_daily.reset_index(inplace=True)
    return var_df_daily


def cell_assimilation(lat_idx, lon_idx):

    save_ensemble = cfg.save_ensemble
    real_time_restart = cfg.real_time_restart

    if save_ensemble:
        name_ensemble = "ensbl_" + str(lat_idx) +\
            "_" + str(lon_idx) + ".pkl.blp"
        name_ensemble = os.path.join(cfg.save_ensemble_path, name_ensemble)

    if real_time_restart:
        name_restart = "init_" + str(lat_idx) +\
            "_" + str(lon_idx) + ".pkl.blp"
        name_restart = os.path.join(cfg.real_time_restart_path, name_restart)

    if cfg.load_prev_run:
        filename = ("Reconstructed_cell_" + str(lat_idx)
                    + "_" + str(lon_idx) + ".pkl.blp")
    else:
        filename = ("cell_" + str(lat_idx) + "_" + str(lon_idx) + ".pkl.blp")

    filename = os.path.join(cfg.output_path, filename)

    if cfg.write_stat_full:
        stat_name_list = ['min', 'max', 'Q1', 'Q3', 'median', 'mean', 'std']
    else:
        stat_name_list = ['mean', 'std']

    # Check if file allready exist if is a restart run
    if (cfg.restart_run and os.path.exists(filename)):
        return None

    dates_obs = ifn.get_dates_obs()
    observations, errors = ifn.obs_array(dates_obs, lat_idx, lon_idx)

    if isinstance(observations, str):  # check if masked
        return None

    main_forcing = model.forcing_table(lat_idx, lon_idx)

    if ifn.forcing_check(main_forcing):
        print("NA's found in: " + str(lat_idx) + "," + str(lon_idx))
        return None

    time_dict = ifn.simulation_steps(observations, dates_obs)

    # If no obs in the cell or det_OL, run openloop, unless cfg.load_prev_run
    if (np.isnan(observations).all() or
            cfg.da_algorithm == "deterministic_OL") and not cfg.load_prev_run:
        ifn.run_model_openloop(lat_idx, lon_idx, main_forcing, filename)
        return None

    # Inicialice results dataframes
    # TODO: make function
    DA_Results = model.init_result(time_dict["del_t"], DA=True)  # DA parameter
    OL_Sim = model.init_result(
        time_dict["del_t"], OL=True)       # OL simulation
    poste_stat = model.init_result(time_dict["del_t"])
    prior_stat = model.init_result(time_dict["del_t"])

    # initialice Ensemble class
    # open previous ensemble if reastart (not if reconstruct)
    if real_time_restart and not cfg.load_prev_run:
        try:
            Ensemble = ifn.io_read(name_restart)
            Ensemble.real_time_restart = True
        except FileNotFoundError:
            Ensemble = SnowEnsemble(lat_idx, lon_idx)
    else:
        Ensemble = SnowEnsemble(lat_idx, lon_idx)

    # Loop over assimilation steps
    for step in range(len(time_dict["Assimilation_steps"])-1):

        # subset forcing, errors and observations
        observations_sbst = observations[time_dict["Assimilation_steps"][step]:
                                         time_dict["Assimilation_steps"][step
                                                                         + 1]]
        error_sbst = errors[time_dict["Assimilation_steps"][step]:
                            time_dict["Assimilation_steps"][step
                                                            + 1]]

        forcing_sbst = main_forcing[time_dict["Assimilation_steps"][step]:
                                    time_dict["Assimilation_steps"][step + 1]]\
            .copy()

        Ensemble.create(forcing_sbst, observations_sbst, error_sbst, step)

        # store prior ensemble
        model.store_sim(prior_stat,
                        Ensemble, time_dict, step, save_prior=True)

        if cfg.load_prev_run:
            # HACK: Simply, store the prior before any DA to reconstruct
            # and exit

            # Write results
            cell_data = {"Ensemble_mean": prior_stat['mean'],
                         "Ensemble_sd":  prior_stat['std']}
            ifn.io_write(filename, ifn.downcast_output(cell_data))
            return None

        with threadpool_limits(limits=cfg.numpy_threads):
            step_results = flt.implement_assimilation(Ensemble, step)
        # Store results in dataframesprior_mean

        model.storeDA(DA_Results, step_results, observations_sbst, error_sbst,
                      time_dict, step)

        model.store_sim(poste_stat, Ensemble, time_dict, step)

        if cfg.da_algorithm in ['IES-MCMC', 'IES-MCMC_AI']:
            mcmc_stat = model.store_sim(Ensemble, time_dict, step, MCMC=True)

        # If redraw, calculate the postrior shape
        if cfg.redraw_prior:
            Ensemble.posterior_shape()

        # Resample if filtering
        if cfg.da_algorithm in ["PF", "PBS"]:
            Ensemble.resample(step_results["resampled_particles"])
        # Optionally, creating new parameters per season (after resampling)
        if cfg.da_algorithm in ["PBS", "ProPBS", "AdaPBS", "ES",
                                "IES", "PIES", "IES-MCMC"]:
            Ensemble.season_rejuvenation()

    # Store OL
    model.storeOL(OL_Sim, Ensemble, observations_sbst,
                  time_dict, step)

    # Save some space
    if cfg.write_stat_daily:
        poste_stat = {key: mean_daily(
            poste_stat[key]) for key in stat_name_list}
        prior_stat = {key: mean_daily(
            prior_stat[key]) for key in stat_name_list}

    # Write results
    cell_data = {**{"DA_Results": DA_Results, "OL_Sim": OL_Sim},
                 **{key+'_Prior': prior_stat[key] for key in stat_name_list},
                 **{key+'_Post': poste_stat[key] for key in stat_name_list}, }

    if cfg.da_algorithm in ['IES-MCMC', 'IES-MCMC_AI']:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mcmc_stat = {key:
                         mcmc_stat[key] for key in
                         stat_name_list}

        cell_data['mcmc_stat'] = {**cell_data, **mcmc_stat}

    # downcast and write output
    ifn.io_write(filename, ifn.downcast_output(cell_data))

    # Save ensemble
    if (save_ensemble):
        ifn.io_write(name_ensemble, Ensemble)

    if real_time_restart:  # save space
        Ensemble.forcing = []  # save some space
        Ensemble.origin_state = pd.DataFrame()
        Ensemble.state_membres = [0 for i in range(Ensemble.members)]
        ifn.io_write(name_restart, Ensemble)
