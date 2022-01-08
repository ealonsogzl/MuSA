#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal functions to read and tidy the forcing, as well as launch the real
assimilation functions.

Author: Esteban Alonso González - e.alonsogzl@gmail.com
"""
import glob
import os
import shutil
import datetime as dt
import netCDF4 as nc
import numpy as np
import pandas as pd
import modules.fsm_tools as fsm
import modules.filters as flt
from modules.internal_class import SnowEnsemble
import config as cfg
import constants as cnt

if cfg.save_ensemble:
    import pickle
    import lzma
    import copy


def obs_array(lat_idx, lon_idx):

    nc_obs_path = cfg.nc_obs_path
    nc_maks_path = cfg.nc_maks_path
    obs_var_names = cfg.obs_var_names
    dates_obs = cfg.dates_obs
    date_ini = cfg.date_ini
    date_end = cfg.date_end

    dates_obs.sort()
    dates_obs = np.asarray([dt.datetime.strptime(date, "%Y-%m-%d %H:%M")
                            for date in dates_obs])

    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
    del_t = np.asarray([date_ini + dt.timedelta(hours=n)
                        for n in range(int((date_end - date_ini).days
                                           * 24 + 24))])

    obs_idx = np.searchsorted(del_t, dates_obs)

    files = glob.glob(nc_obs_path + "*.nc")
    # TODO: let the user define the prefix of the observations
    files.sort()

    if len(files) != len(dates_obs):  # check if num of dates == num of files
        raise Exception("Number of dates different of number of obs files")

    mask = glob.glob(nc_maks_path + "*.nc")

    if mask:  # If mask exists, return string if masked
        mask = nc.Dataset(mask[0])
        mask_value = mask.variables['mask'][lat_idx, lon_idx]
        mask.close()
        if np.ma.is_masked(mask_value):
            array_obs = "Out_of_AOI"
            return array_obs

    # Initialize obs matrix
    obs_matrix = np.empty((len(del_t), len(obs_var_names)))

    for cont, obs_var in enumerate(obs_var_names):

        array_obs = np.empty(len(del_t))
        array_obs[:] = np.nan

        tmp_storage = []

        for i, ncfile in enumerate(files):

            data_temp = nc.Dataset(ncfile)
            nc_value = data_temp.variables[obs_var][:, lat_idx, lon_idx]
            # Check if masked
            # TODO: Check if there is a better way to do this
            if np.ma.is_masked(nc_value):
                nc_value = np.ma.getdata(nc_value)
                nc_value[:] = np.nan
            else:
                nc_value = np.ma.getdata(nc_value)

            tmp_storage.extend(nc_value)
            data_temp.close()

        array_obs[obs_idx] = tmp_storage

        obs_matrix[:, cont] = array_obs

    # Remove extra dimension when len(obs_var_names) == 1
    obs_matrix = np.squeeze(obs_matrix)

    return obs_matrix


def nc_array_forcing(nc_forcing_path, lat_idx, lon_idx, nc_var_name,
                     date_ini, date_end):
    """
    Extract an array of forcing timesteps from an nc
    Parameters
    ----------
    nc_forcing_path : string
        Path of the netcdf.
    lat_idx : int
        Netcdf latitude idx.
    lon_idx : int
        Netcdf longitude idx.
    nc_var_name : string
        Netcdf variable name.
    date_ini : string
        First date of the forcing.
    date_end : string
        Last of the end of the forcing

    Returns
    -------
    array_nc : np array
        Array of forcing timesteps.
    """

    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
    del_t = np.asarray([date_ini + dt.timedelta(hours=n)
                        for n in range(int((date_end -
                                            date_ini).days*24 + 24))])

    files = glob.glob(nc_forcing_path + "*.nc")
    files.sort()

    # TODO: Bug reported somewhere. Test with a different version of netcdf
    # Surprisingly high RAM consumption, netCDF4 bug?
    # nc_data = nc.MFDataset(files)
    # array_nc = nc_data.variables[nc_var_name][:,lat_idx,lon_idx]
    # nc_data.close()

    array_nc = []
    # TODO: preallocate the array and fill, array_obs = np.empty(len(array_nc))

    for ncfile in files:
        data_temp = nc.Dataset(ncfile)
        array_temp = data_temp.variables[nc_var_name][:, lat_idx, lon_idx]
        array_temp = np.ma.getdata(array_temp)
        array_nc.extend(array_temp)
        data_temp.close()

    array_nc = np.array(array_nc)

    if len(del_t) != len(array_nc):
        raise Exception("date_end - date_ini longuer than forcing")

    return array_nc


def forcing_table(lat_idx, lon_idx):

    nc_forcing_path = cfg.nc_forcing_path
    frocing_var_names = cfg.frocing_var_names
    date_ini = cfg.date_ini
    date_end = cfg.date_end
    intermediate_path = cfg.intermediate_path

    # Path to intermediate file
    final_directory = os.path.join(intermediate_path,
                                   (str(lon_idx) + "_" +
                                    str(lat_idx) + ".pkl"))

    # try to read the forcing from a dumped file
    if os.path.exists(final_directory):

        forcing_df = pd.read_pickle(final_directory)

    else:

        short_w = nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                   frocing_var_names["SW_var_name"],
                                   date_ini, date_end)

        long_wave = nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                     frocing_var_names["LW_var_name"],
                                     date_ini, date_end)

        prec = nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                frocing_var_names["Precip_var_name"],
                                date_ini, date_end)

        temp = nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                frocing_var_names["Temp_var_name"],
                                date_ini, date_end)

        rel_humidity = nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                        frocing_var_names["RH_var_name"],
                                        date_ini, date_end)

        wind = nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                frocing_var_names["Wind_var_name"],
                                date_ini, date_end)

        press = nc_array_forcing(nc_forcing_path, lat_idx, lon_idx,
                                 frocing_var_names["Press_var_name"],
                                 date_ini, date_end)

        date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
        date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
        del_t = [date_ini + dt.timedelta(hours=n)
                 for n in range(int((date_end - date_ini).days*24 + 24))]

        forcing_df = pd.DataFrame({"year": del_t,
                                  "month": del_t,
                                   "day": del_t,
                                   "hours": del_t,
                                   "SW": short_w,
                                   "LW": long_wave,
                                   "Prec": prec,
                                   "Ta": temp,
                                   "RH": rel_humidity,
                                   "Ua": wind,
                                   "Ps": press})

        forcing_df["year"] = forcing_df["year"].dt.year
        forcing_df["month"] = forcing_df["month"].dt.month
        forcing_df["day"] = forcing_df["day"].dt.day
        forcing_df["hours"] = forcing_df["hours"].dt.hour

        # write intermediate file to avoid re-reading the nc files
        forcing_df.to_pickle(final_directory)

        if len(del_t) != len(forcing_df.index):
            raise Exception("date_end - date_ini longuer than forcing")

    return forcing_df


def forcing_check(forcing_df):

    if forcing_df.isnull().values.any():
        return True

    else:
        return False


def unit_conversion(forcing_df):

    forcing_offset = cnt.forcing_offset
    forcing_multiplier = cnt.forcing_multiplier

    forcing_df.SW = forcing_df.SW * forcing_multiplier["SW"]
    forcing_df.LW = forcing_df.LW * forcing_multiplier["LW"]
    forcing_df.Prec = forcing_df.Prec * forcing_multiplier["Prec"]
    forcing_df.Ta = forcing_df.Ta * forcing_multiplier["Ta"]
    forcing_df.RH = forcing_df.RH * forcing_multiplier["RH"]
    forcing_df.Ua = forcing_df.Ua * forcing_multiplier["Ua"]
    forcing_df.Ps = forcing_df.Ps * forcing_multiplier["Ps"]

    forcing_df.SW = forcing_df.SW + forcing_offset["SW"]
    forcing_df.LW = forcing_df.LW + forcing_offset["LW"]
    forcing_df.Prec = forcing_df.Prec + forcing_offset["Prec"]
    forcing_df.Ta = forcing_df.Ta + forcing_offset["Ta"]
    forcing_df.RH = forcing_df.RH + forcing_offset["RH"]
    forcing_df.Ua = forcing_df.Ua + forcing_offset["Ua"]
    forcing_df.Ps = forcing_df.Ps + forcing_offset["Ps"]

    return(forcing_df)


def nc_idx():

    lon = cfg.aws_lon
    lat = cfg.aws_lat
    nc_forcing_path = cfg.nc_forcing_path
    forcing_dim_names = cfg.forcing_dim_names

    files = glob.glob(nc_forcing_path + "*.nc")
    files.sort()

    data = nc.Dataset(files[0])

    lat_name_var = forcing_dim_names["lat_forz_var_name"]
    lon_name_var = forcing_dim_names["lon_forz_var_name"]
    lats = data.variables[lat_name_var][:]
    lons = data.variables[lon_name_var][:]

    lat_idx = (np.abs(lats - lat)).argmin()
    lon_idx = (np.abs(lons - lon)).argmin()

    return lon_idx, lat_idx


def get_dims():

    nc_forcing_path = cfg.nc_forcing_path
    forcing_dim_names = cfg.forcing_dim_names

    example_file = glob.glob(nc_forcing_path + "*.nc")[0]
    example_file = nc.Dataset(example_file)

    lat_name_var = forcing_dim_names["lat_forz_var_name"]
    lon_name_var = forcing_dim_names["lon_forz_var_name"]
    n_lats = len(example_file.variables[lat_name_var][:])
    n_lons = len(example_file.variables[lon_name_var][:])
    return n_lons, n_lats


def expand_grid():

    nc_maks_path = cfg.nc_maks_path

    n_lons, n_lats = get_dims()
    grid = np.meshgrid(range(n_lons), range(n_lats))
    grid = np.array(grid).reshape(2, n_lons * n_lats).T

    mask = glob.glob(nc_maks_path + "*.nc")

    if mask:  # If mask exists, return string if masked
        mask = nc.Dataset(mask[0])
        mask_value = mask.variables['mask'][:, :]
        mask.close()
        mask = mask_value.flatten('C')

        grid = grid[~mask.mask]

    return grid


def simulation_steps(observations):

    date_ini = cfg.date_ini
    date_end = cfg.date_end
    dates_obs = cfg.dates_obs
    season_ini_day = cfg.season_ini_day
    season_ini_month = cfg.season_ini_month,
    assimilation_strategy = cfg.assimilation_strategy

    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")

    dates_obs.sort()
    dates_obs = [dt.datetime.strptime(date, "%Y-%m-%d %H:%M")
                 for date in dates_obs]

    del_t = np.asarray([date_ini + dt.timedelta(hours=x)
                        for x in range(int((date_end-date_ini).total_seconds()
                                           / 3600) + 1)])

    obs_idx = np.searchsorted(del_t, dates_obs)

    # Remove observation NaNs from simulations steps
    if observations.ndim == 1:
        obs_values = observations[obs_idx]
        check = ~np.isnan(obs_values)
        obs_idx = obs_idx[check]
    else:
        obs_values = observations[obs_idx, :]
        check = np.all(~np.isnan(obs_values), axis=1)
        obs_idx = obs_idx[check]

    days = [date.day for date in del_t]
    months = [date.month for date in del_t]
    hours = [date.hour for date in del_t]

    season_ini_cuts = np.argwhere((np.asarray(days) == season_ini_day) &
                                  (np.asarray(months) == season_ini_month) &
                                  (np.asarray(hours) == 0))

    if assimilation_strategy == "smoothing":
        assimilation_steps = season_ini_cuts[:, 0]
    elif (assimilation_strategy == "filtering" or
          assimilation_strategy == "direct_insertion"):
        # HACK: I add one to easy the subset of the forcing
        assimilation_steps = obs_idx + 1
    else:
        raise Exception("Choose between smoothing or filtering")

    lng_del_t = np.asarray(len(del_t))
    assimilation_steps = np.append(0, assimilation_steps)
    assimilation_steps = np.append(assimilation_steps, lng_del_t)

    return {"del_t": del_t,
            "obs_idx": obs_idx,
            "Assimilaiton_steps": assimilation_steps}


def store_OL_vars(Result_df, Ensemble):

    var_to_assim = cfg.var_to_assim

    # Store SWE and SD
    Result_df.loc[:, "SWE_origin"] =\
        Ensemble.origin_state.iloc[:, 5].to_numpy()
    Result_df.loc[:, "SD_origin"] =\
        Ensemble.origin_state.iloc[:, 4].to_numpy()

    # Store OL vars

    OL_state = Ensemble.origin_state.copy()

    for var in var_to_assim:

        assim_idx = fsm.get_var_state_position(var)
        OL_tmp = OL_state.iloc[:, assim_idx].to_numpy()

        Result_df.loc[:, var + "_ol"] = OL_tmp


def result_to_df(Result_df, step_results, observations_sbst, time_dict, step):

    vars_to_perturbate = cfg.vars_to_perturbate
    var_to_assim = cfg.var_to_assim

    rowIndex = Result_df.index[time_dict["Assimilaiton_steps"][step]:
                               time_dict["Assimilaiton_steps"][step + 1]]

    if len(var_to_assim) > 1:
        for i, var in enumerate(var_to_assim):
            Result_df.loc[rowIndex, var] = observations_sbst[:, i]
    else:
        var = var_to_assim[0]
        Result_df.loc[rowIndex, var] = observations_sbst

    Result_df.loc[rowIndex, "SWE_ens_mean"] = step_results["SWE_ens_mean"]
    Result_df.loc[rowIndex, "SD_ens_mean"] = step_results["SD_ens_mean"]
    Result_df.loc[rowIndex, "SWE_ens_sd"] = step_results["SWE_ens_sd"]
    Result_df.loc[rowIndex, "SD_ens_sd"] = step_results["SD_ens_sd"]
    Result_df.loc[rowIndex, "SWE_assim_mean"] = step_results["SWE_assim_mean"]
    Result_df.loc[rowIndex, "SD_assim_mean"] = step_results["SD_assim_mean"]
    Result_df.loc[rowIndex, "SWE_assim_sd"] = step_results["SWE_assim_sd"]
    Result_df.loc[rowIndex, "SD_assim_sd"] = step_results["SD_assim_sd"]

    # Add perturbation parameters to Results
    for var_p in vars_to_perturbate:
        Result_df.loc[rowIndex, var_p +
                      "_noise_mean"] = step_results[var_p + "_noise_mean"]
        Result_df.loc[rowIndex, var_p +
                      "_noise_sd"] = step_results[var_p + "_noise_sd"]

    # Add posterior assimilated vars
    post_vars = step_results["post_vars"]

    for count, var_p in enumerate(var_to_assim):
        Result_df.loc[rowIndex, var_p +
                      "_posterior"] = post_vars[count]


def init_result(del_t):

    var_to_assim = cfg.var_to_assim
    vars_to_perturbate = cfg.vars_to_perturbate
    vars_to_perturbate = [x + "_noise" for x in vars_to_perturbate]

    posterior_vars = [x + "_posterior" for x in var_to_assim]
    ol_vars = [x + "_ol" for x in var_to_assim]
    # Concatenate
    col_names = ["Date", "SWE_ens_mean",
                 "SD_ens_mean", "SWE_ens_sd", "SD_ens_sd",
                 "SWE_assim_mean", "SD_assim_mean",
                 "SWE_assim_sd", "SD_assim_sd",
                 "SWE_origin", "SD_origin"] + var_to_assim + ol_vars +\
        posterior_vars + vars_to_perturbate

    # Create results dataframe
    Results = pd.DataFrame(np.nan, index=range(len(del_t)), columns=col_names)

    Results["Date"] = [x.strftime('%d/%m/%Y-%H:%S') for x in del_t]

    return Results


def run_FSM_openloop(lon_idx, lat_idx, main_forcing, temp_dest, filename):

    print("No observations in: " + str(lon_idx) + "," + str(lat_idx))
    real_forcing = main_forcing.copy()
    fsm.fsm_forcing_wrt(real_forcing, temp_dest)
    fsm.write_init(temp_dest)
    fsm.fsm_run(temp_dest)
    state = fsm.fsm_read_output(temp_dest, read_dump=False)
    state.columns = ["year", "month", "day", "hour", "snd", "SWE", "Sveg",
                     "1Tsoil", "2Tsoil", "3Tsoil", "4Tsoil", "Tsrf", "Tveg"]

    state.to_csv(filename, sep=",", header=True, index=False)


def cell_assimilation(lon_idx, lat_idx):

    save_ensemble = cfg.save_ensemble

    filename = ("Result_" + str(lon_idx) + "_" + str(lat_idx) + ".csv")
    filename = os.path.join(cfg.output_path, filename)

    # Check if file allready exist
    if os.path.exists(filename):
        return None

    observations = obs_array(lat_idx, lon_idx)

    if isinstance(observations, str):  # check if masked
        return None

    temp_dest = fsm.fsm_copy(lon_idx, lat_idx)

    main_forcing = forcing_table(lat_idx, lon_idx)

    if forcing_check(main_forcing):
        print("NA's found in: " + str(lon_idx) + "," + str(lat_idx))
        return None

    main_forcing = unit_conversion(main_forcing)

    time_dict = simulation_steps(observations)

    # If no obs in the cell, run openloop
    if np.isnan(observations).all():
        run_FSM_openloop(lon_idx, lat_idx, main_forcing, temp_dest, filename)

    # Inicialice results dataframe
    Results = init_result(time_dict["del_t"])

    # initialice Ensemble class
    Ensemble = SnowEnsemble(temp_dest)

    # Initialice Ensemble list if enabled in cfg
    if save_ensemble:
        ensemble_list = []

    # Loop over assimilation steps
    for step in range(len(time_dict["Assimilaiton_steps"])-1):

        # subset forcing and observations
        observations_sbst = observations[time_dict["Assimilaiton_steps"][step]:
                                         time_dict["Assimilaiton_steps"][step
                                                                         + 1]]

        forcing_sbst = main_forcing[time_dict["Assimilaiton_steps"][step]:
                                    time_dict["Assimilaiton_steps"][step + 1]]\
            .copy()

        Ensemble.create(forcing_sbst, step)

        step_results = flt.implement_assimilation(Ensemble, observations_sbst,
                                                  step, forcing_sbst)
        if save_ensemble:
            # deepcopy necesary to not to change all
            Ensemble_tmp = copy.deepcopy(Ensemble)
            ensemble_list.append(Ensemble_tmp)

        # Resample if filtering
        if(cfg.assimilation_strategy == "filtering" and
           "resampled_particles" in step_results):
            Ensemble.resample(step_results["resampled_particles"])

        # Store values in Results in df ifnot direct insertion
        if step_results is not None:
            result_to_df(Results, step_results, observations_sbst,
                         time_dict, step)

    # Store open loop SWE, SD and assimilated vars
    store_OL_vars(Results, Ensemble)

    # TODO: create a write function with NCDF support
    Results.to_csv(filename, sep=",", header=True, index=False,
                   float_format="%.3f")

    # Save ensemble
    if save_ensemble:
        name_ensemble = "ensbl_" + str(lon_idx) + "_" + str(lat_idx) + ".xz"
        name_ensemble = os.path.join(cfg.save_ensemble_path, name_ensemble)
        filehandler = lzma.open(name_ensemble, 'wb')
        pickle.dump(ensemble_list, filehandler)

    # Clean tmp directory
    shutil.rmtree(temp_dest, ignore_errors=True)


def open_loop_simulation(lon_idx, lat_idx):

    nc_obs_path = cfg.nc_obs_path

    # Check if file allready exist
    filename = ("Result_openloop_" + str(lat_idx) +
                "_" + str(lon_idx) + ".csv")
    filename = os.path.join(cfg.output_path, filename)

    if os.path.exists(filename):
        return None

    mask = glob.glob(nc_obs_path + "*mask*.nc")

    if mask:  # If mask exists, check if the cell is masked
        mask = nc.Dataset(mask[0])
        mask_value = mask.variables['mask'][lat_idx, lon_idx]
        mask.close()
        if np.ma.is_masked(mask_value):
            return None

    temp_dest = fsm.fsm_copy(lon_idx, lat_idx)

    main_forcing = forcing_table(lat_idx, lon_idx)

    main_forcing = unit_conversion(main_forcing)

    real_forcing = main_forcing.copy()
    fsm.fsm_forcing_wrt(real_forcing, temp_dest)
    fsm.write_init(temp_dest)
    fsm.fsm_run(temp_dest)
    state, flux, dump = fsm.fsm_read_output(temp_dest, read_flux=True)

    # Join fluxes to state variables
    flux_temp = flux.iloc[:, 4:11].copy()
    Results = state.join(flux_temp).copy()

    # TODO: create a write function with NCDF support
    Results.to_csv(filename, sep=",", header=True, index=False,
                   float_format="%.3f")
    # Clean tmp directory
    shutil.rmtree(temp_dest, ignore_errors=True)
