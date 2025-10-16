#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some functions to interact with Snow17.

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import os
import constants as cnt
import numpy as np
import datetime as dt
import pandas as pd
import config as cfg
import copy
import pdcast as pdc
import warnings
import modules.internal_fns as ifn
from statsmodels.stats.weightstats import DescrStatsW
from modules.snow17 import snow17
from datetime import datetime, timedelta
if cfg.DAsord:
    from modules.user_optional_fns import snd_ord

if cfg.DAsord:
    model_columns = ("SWE", "snd", "outflow", tuple(cfg.DAord_names))
else:
    model_columns = ("SWE", "snd", "outflow")


def prepare_forz(forcing_sbst):  # time, prec, tair, p_atm, lat

    year = forcing_sbst['year'].values
    month = forcing_sbst['month'].values
    day = forcing_sbst['day'].values
    hours = forcing_sbst['hours'].values

    time = np.arange(datetime(year[0], month[0], day[0], hours[0]),
                     datetime(year[-1], month[-1], day[-1], hours[-1])
                     + timedelta(hours=1),
                     timedelta(hours=1)).astype(datetime)

    time_jday = np.array([x.timetuple()[-2] for x in time])

    Temp = forcing_sbst['Ta'].values
    prec = forcing_sbst["Prec"].values
    p_atm = forcing_sbst["Prec"].values
    lat = forcing_sbst["XLAT"].values

    return time_jday, prec*3600, Temp-cnt.KELVING_CONVER, p_atm*0.01, lat


def model_run(forcing_sbst, init=None):

    time_jday, prec, tair, p_atm, lat = prepare_forz(forcing_sbst)

    if init is None:
        init = np.zeros(5)

    SWE, snd, outflow, init = snow17(time_jday, prec, tair, p_atm, lat, init)

    Results = pd.DataFrame({'SWE': SWE,
                            'snd': snd,
                            'outflow': outflow})

    # add optional variables
    if cfg.DAsord:
        Results = snd_ord(Results)

    return Results, init


def model_copy(y_id, x_id):
    return None


def write_nlst(temp_dest, step):
    return None


def model_compile():
    return None


def model_compile_HPC(HPC_task_id):
    return None


def model_read_output(fsm_path, read_dump=True):
    return None


def model_remove(fsm_path):
    return None


def write_input(fsm_path, fsm_input_data):
    return None


def stable_forcing(forcing_df):

    return None


def model_forcing_wrt(forcing_df, temp_dest, step):

    return None


def write_dump(dump, fsm_path):
    return None


def get_var_state_position(var):

    state_columns = model_columns

    return state_columns.index(var)


def storeDA(Result_df, step_results, observations_sbst, error_sbst,
            time_dict, step):

    vars_to_perturbate = cfg.vars_to_perturbate
    var_to_assim = cfg.var_to_assim
    error_names = cfg.obs_error_var_names

    rowIndex = Result_df.index[time_dict["Assimilation_steps"][step]:
                               time_dict["Assimilation_steps"][step + 1]]

    if len(var_to_assim) > 1:
        for i, var in enumerate(var_to_assim):
            Result_df.loc[rowIndex, var] = observations_sbst[:, i]
            Result_df.loc[rowIndex, error_names[i]] = error_sbst[:, i]
    else:
        var = var_to_assim[0]
        Result_df.loc[rowIndex, var] = observations_sbst
        Result_df.loc[rowIndex, error_names] = error_sbst

    # Add perturbation parameters to Results
    for var_p in vars_to_perturbate:
        Result_df.loc[rowIndex, var_p +
                      "_noise_mean"] = step_results[var_p + "_noise_mean"]
        Result_df.loc[rowIndex, var_p +
                      "_noise_sd"] = step_results[var_p + "_noise_sd"]


def storeOL(OL_FSM, Ensemble, observations_sbst, time_dict, step):

    ol_data = Ensemble.origin_state.copy()

    # Store colums
    for n, name_col in enumerate(ol_data.columns):
        OL_FSM[name_col] = ol_data.iloc[:, [n]].to_numpy()


def store_sim( Ensemble, time_dict, step, MCMC=False, save_prior=False):
    
    if cfg.write_stat_full:
        stat_name_list = ['min', 'max', 'Q1', 'Q3', 'median', 'mean', 'std']
    else:
        stat_name_list = ['mean', 'std']
        
    sim_stat = {key: init_result(time_dict["del_t"]) for key in stat_name_list}
    
    if MCMC:
        list_state = copy.deepcopy(Ensemble.state_members_mcmc)
    else:
        list_state = copy.deepcopy(Ensemble.state_membres)

    rowIndex = sim_stat['mean'].index[time_dict["Assimilation_steps"][step]:
                                 time_dict["Assimilation_steps"][step + 1]]
    # Get updated columns
    if save_prior:
        pesos = np.ones_like(Ensemble.wgth)
    else:
        pesos = Ensemble.wgth

    for n, name_col in enumerate(list(list_state[0].columns)):

        # create matrix of colums
        col_arr = [list_state[x].iloc[:, n].to_numpy()
                   for x in range(len(list_state))]
        col_arr = np.vstack(col_arr)
        
        d1 = DescrStatsW(col_arr, weights=pesos)

        if len( sim_stat.keys()) == 2: # Mean, Std 
            sim_stat['mean'].loc[rowIndex, name_col] = d1.mean
            sim_stat['std'].loc[rowIndex, name_col] = d1.std
        else:  
            perc = d1.quantile([ 0, 0.25, 0.5, 0.75, 1 ]).values
            sim_stat['min'].loc[rowIndex, name_col] = perc[0,:]
            sim_stat['Q1'].loc[rowIndex, name_col] = perc[1,:]
            sim_stat['median'].loc[rowIndex, name_col] = perc[2,:]
            sim_stat['Q3'].loc[rowIndex, name_col] = perc[3,:]
            sim_stat['max'].loc[rowIndex, name_col] = perc[4,:]
            sim_stat['mean'].loc[rowIndex, name_col] = d1.mean
            sim_stat['std'].loc[rowIndex, name_col] = d1.std
    return sim_stat

def init_result(del_t, DA=False):

    if DA:
        # Concatenate
        col_names = ["Date"]

        # Create results dataframe
        Results = pd.DataFrame(np.nan, index=range(len(del_t)),
                               columns=col_names)

        Results["Date"] = [x.strftime('%d/%m/%Y-%H:%S') for x in del_t]
        return Results

    else:
        # Create results dataframe
        Results = pd.DataFrame(np.nan, index=range(len(del_t)),
                               columns=model_columns)

        Results["Date"] = [x.strftime('%d/%m/%Y-%H:%S') for x in del_t]

        return Results


def forcing_table(lat_idx, lon_idx, step=0):

    nc_forcing_path = cfg.nc_forcing_path
    forcing_var_names = cfg.forcing_var_names
    param_var_names = cfg.param_var_names
    date_ini = cfg.date_ini
    date_end = cfg.date_end
    intermediate_path = cfg.intermediate_path

    # Path to intermediate file
    final_directory = os.path.join(intermediate_path,
                                   (str(lat_idx) + "_" +
                                    str(lon_idx) + ".pkl"))

    # try to read the forcing from a dumped file
    if os.path.exists(final_directory) and (cfg.restart_forcing or
                                            (cfg.implementation ==
                                             "Spatial_propagation" and
                                             step != 0)):

        forcing_df = ifn.io_read(final_directory)

    else:

        prec = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                    forcing_var_names["Precip_var_name"],
                                    date_ini, date_end)

        temp = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                    forcing_var_names["Temp_var_name"],
                                    date_ini, date_end)

        press = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                     forcing_var_names["Press_var_name"],
                                     date_ini, date_end)
        try:
            XLAT = ifn.nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        param_var_names["XLAT_var_name"],
                                        date_ini, date_end)
        except KeyError:
            XLAT = np.repeat(cnt.aprox_lat, len(prec))

        date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
        del_t = ifn.generate_dates(date_ini, date_end)

        forcing_df = pd.DataFrame({"year": del_t,
                                  "month": del_t,
                                   "day": del_t,
                                   "hours": del_t,
                                   "Prec": prec,
                                   "Ta": temp,
                                   "press": press,
                                   "XLAT": XLAT})

        forcing_df["year"] = forcing_df["year"].dt.year
        forcing_df["month"] = forcing_df["month"].dt.month
        forcing_df["day"] = forcing_df["day"].dt.day
        forcing_df["hours"] = forcing_df["hours"].dt.hour

        forcing_df = unit_conversion(forcing_df)

        if len(del_t) != len(forcing_df.index):
            raise Exception("date_end - date_ini longuer than forcing")

        # write intermediate file to avoid re-reading the nc files
        if cfg.save_int_forcing:
            ifn.io_write(final_directory, forcing_df)

    return forcing_df


def unit_conversion(forcing_df):

    forcing_offset = cnt.forcing_offset
    forcing_multiplier = cnt.forcing_multiplier

    forcing_df.Prec = forcing_df.Prec * forcing_multiplier["Prec"]
    forcing_df.Ta = forcing_df.Ta * forcing_multiplier["Ta"]

    forcing_df.Prec = forcing_df.Prec + forcing_offset["Prec"]
    forcing_df.Ta = forcing_df.Ta + forcing_offset["Ta"]

    # Save some space
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        forcing_df = pdc.downcast(forcing_df,
                                  numpy_dtypes_only=True)

    return forcing_df
