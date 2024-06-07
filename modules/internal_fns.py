#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal functions to read and tidy the forcing, as well as launch the real
assimilation functions.

Author: Esteban Alonso González - alonsoe@ipe.csic.es
"""

import glob
import os
import shutil
import datetime as dt
import netCDF4 as nc
import numpy as np
import multiprocessing as mp
import pandas as pd
import config as cfg
if cfg.numerical_model == 'FSM2':
    import modules.fsm_tools as model
elif cfg.numerical_model == 'dIm':
    import modules.dIm_tools as model
elif cfg.numerical_model == 'snow17':
    import modules.snow17_tools as model
else:
    raise Exception('Model not implemented')
import pickle
import blosc
import warnings
import pdcast as pdc
if cfg.MPI:
    from mpi4py.futures import MPIPoolExecutor
import re
import subprocess
import gc
import time


def pre_cheks():
    """
    This function is a kind of helper, which tries to find problems in
    the configuration (it will be improved with time).
    """
    if cfg.load_prev_run and cfg.implementation == 'Spatial_propagation':
        raise Exception('Disable Spatial_propagation if load_prev_run is '
                        ' enabled, even considering that load_prev_run '
                        'supports simulations generated from '
                        'Spatial_propagation simulation.')
    if cfg.timeout and cfg.MPI:
        warnings.warn("timeout is ignored  MPI")


def last_line(filename):
    with open(filename, 'r') as file:
        lineas = file.readlines()
        if lineas:
            return lineas[-1]
        else:
            return None


def return_step_j(logfile):
    try:
        # Leer la última línea
        ultima_linea = last_line(logfile)

        # Si la última línea existe, extraer los valores de step y j
        if ultima_linea:
            # Usar una expresión regular para extraer los valores
            match = re.search(r'step:\s*(\d+)\s*-\s*j:\s*(\d+)', ultima_linea)
            if match:
                step = int(match.group(1))
                j = int(match.group(2))
        else:
            # log file empty or innexsitent
            step = 0
            j = 0

        return step, j
    except Exception:
        step = 0
        j = 0
        print('Not possible to restart, check spatiallogfile.txt for errors.',
              'Starting simulation from the beginning')
        return step, j


def change_chunk_size_nccopy(input_file):
    # Open the input NetCDF file to get dimension sizes
    with nc.Dataset(input_file, 'r') as dataset:
        # Determine chunk sizes based on dimensions
        chunk_sizes = {dim: min(50, len(dataset.dimensions[dim]))
                       for dim in dataset.dimensions}

    # Build the nccopy command to copy to a temporary file
    temp_file = input_file + '.temp'
    command = ["nccopy", "-c", ",".join(f"{dim}/{size}"
                                        for dim, size in chunk_sizes.items()),
               input_file, temp_file]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print("Chunking successful.")

        # Replace the original file with the temporary file
        os.remove(input_file)
        os.rename(temp_file, input_file)
        print("File replaced successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Not possible to chunk file: {input_file}."
              "Manual chunking is recommended")


def io_write(filename, obj):
    # TODO: Explore more compression options
    with open(filename, "wb") as f:
        pickled_data = pickle.dumps(obj)
        compressed_pickle = blosc.compress(pickled_data)
        f.write(compressed_pickle)


def io_read(filename):
    with open(filename, "rb") as f:
        compressed_pickle = f.read()
        depressed_pickle = blosc.decompress(compressed_pickle)
        obj = pickle.loads(depressed_pickle)
        return obj


def reduce_size_state(df_state, observations):

    var_to_assim = cfg.var_to_assim
    df_state = df_state.copy()

    for count, col in enumerate(df_state.columns):

        if col in var_to_assim:
            pos = var_to_assim.index(col)
            mask = np.ones(len(df_state.index), bool)

            if observations.ndim > 1:

                mask[~np.isnan(observations[:, pos])] = 0
            else:
                mask[~np.isnan(observations)] = 0

            df_state.loc[mask, col] = np.nan

        else:

            df_state[col] = np.nan

    return df_state


def downcast_output(output):
    # Save some space
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for n in output.keys():

            output[n] = pdc.downcast(output[n],
                                     numpy_dtypes_only=True)
    return output


def chunker(seq, size):
    """
    Splits a sequence into chunks of a given size.

    Parameters:
    seq (list): The sequence to be split.
    size (int): The size of each chunk.

    Returns:
    list: A list of chunks.
    """
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]


def pool_wrap(func, inputs, nprocess, timeout=None):
    """
    Wraps the multiprocessing pool or MPI pool execution.

    Parameters:
    func (callable): The function to be executed in parallel.
    inputs (list): The input arguments for the function.
    nprocess (int): The number of processes to use.
    timeout (int, optional): The maximum time to wait for the results.

    Raises:
    TimeoutError: If the pool execution times out.
    Exception: If any other error occurs during pool execution.
    """
    if cfg.MPI:
        with MPIPoolExecutor() as pool:

            pool.starmap(func, inputs,  timeout=timeout)

    else:
        pool = mp.Pool(processes=nprocess)
        result = pool.starmap_async(func, inputs)
        try:
            if timeout:
                # Wait until the result is ready or timeout
                result.get(timeout)
            else:
                result.get()

        except Exception:
            # Terminate the pool if an exception occurs
            pool.terminate()
            pool.join()
            gc.collect()
            raise
        finally:
            # Ensure the pool is terminated even if no exception occurs
            pool.terminate()
            pool.join()
            gc.collect()


def safe_pool(func, inputs, nprocess):
    """
    Safely executes a function in parallel, restarting the pool if it freezes.

    Parameters:
    func (callable): The function to be executed in parallel.
    inputs (list): The input arguments for the function.
    nprocess (int): The number of processes to use.
    cells_per_process (int, optional): The multiplier to calculate maximum
    cells per process.
    timeout (int, optional): The maximum time to wait for each chunk.

    This function divides the inputs into chunks and executes them in parallel.
    If the pool freezes, it will restart and retry the execution.
    """
    cells_per_process = cfg.cells_per_process
    timeout = cfg.timeout

    if not cells_per_process:
        cells_per_process = 1

    ncellsmax = cells_per_process * nprocess  # Maximum cells per process

    inputs_chunk = [chunker(x, ncellsmax) for x in inputs]

    for chunk_id in range(len(inputs_chunk[0])):
        chunked_list = [item[chunk_id] for item in inputs_chunk]

        while True:
            chunked_zip = zip(*chunked_list)

            try:
                pool_wrap(func, chunked_zip, nprocess, timeout=timeout)
                break  # Exit the loop if execution is successful
            except mp.context.TimeoutError:
                print("The pool has frozen. Restarting...")

                time.sleep(10)
                pass


def get_dates_obs():

    dates_obs = cfg.dates_obs

    if type(dates_obs) == list:

        dates_obs.sort()
        dates_obs = np.asarray([dt.datetime.strptime(date, "%Y-%m-%d %H:%M")
                                for date in dates_obs])
    elif type(dates_obs) == str:

        dates_obs = pd.read_csv(dates_obs, header=None)
        dates_obs = dates_obs.iloc[:, 0].tolist()
        dates_obs = np.asarray([dt.datetime.strptime(date, "%Y-%m-%d %H:%M")
                                for date in dates_obs])

    else:
        raise Exception('Bad obs date format')

    return dates_obs


def obs_array(dates_obs, lat_idx, lon_idx):

    nc_obs_path = cfg.nc_obs_path
    mask = cfg.nc_maks_path
    obs_var_names = cfg.obs_var_names
    date_ini = cfg.date_ini
    date_end = cfg.date_end
    r_cov = cfg.r_cov

    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
    del_t = generate_dates(date_ini, date_end)

    obs_idx = np.searchsorted(del_t, dates_obs)

    files = glob.glob(nc_obs_path + "*.nc")
    # TODO: let the user define the prefix of the observations
    if len(files) == 0:
        raise Exception('Observation files not found')

    files.sort()

    if mask:  # If mask exists, return string if masked
        mask = nc.Dataset(mask)
        mask_value = mask.variables['mask'][lat_idx, lon_idx]
        mask.close()
        if np.isnan(mask_value):
            array_obs = "Out_of_AOI"
            return array_obs

    # Initialize obs matrix
    obs_matrix = np.empty((len(del_t), len(obs_var_names)))
    error_matrix = np.empty((len(del_t), len(obs_var_names)))

    for cont, obs_var in enumerate(obs_var_names):

        array_obs = np.empty(len(del_t))
        array_obs[:] = np.nan

        array_error = np.empty(len(del_t))
        array_error[:] = np.nan

        tmp_obs_storage = []
        tmp_error_storage = []

        for i, ncfile in enumerate(files):

            data_tmp = nc.Dataset(ncfile)

            if obs_var in data_tmp.variables.keys():

                nc_value = data_tmp.variables[obs_var][:, lat_idx, lon_idx]
                # Check if masked
                # TODO: Check if there is a better way to do this
                if np.ma.is_masked(nc_value):
                    nc_value = nc_value.filled(np.nan)
                else:
                    nc_value = np.ma.getdata(nc_value)

                tmp_obs_storage.extend(nc_value)

                # do the same conditionally for errors

                if r_cov == 'dynamic_error':

                    nc_value = data_tmp.variables[cfg.obs_error_var_names[cont]
                                                  ][:, lat_idx, lon_idx]
                    # Check if masked
                    # TODO: Check if there is a better way to do this
                    if np.ma.is_masked(nc_value):
                        nc_value = nc_value.filled(np.nan)
                    else:
                        nc_value = np.ma.getdata(nc_value)

                    tmp_error_storage.extend(nc_value)
                else:

                    tmp_error_storage = [r_cov[cont]] * len(tmp_obs_storage)
            else:
                tmp_obs_storage.extend([np.nan])
                tmp_error_storage.extend([np.nan])
            data_tmp.close()

        array_obs[obs_idx] = tmp_obs_storage
        array_error[obs_idx] = tmp_error_storage

        obs_matrix[:, cont] = array_obs
        error_matrix[:, cont] = array_error

    # Remove extra dimension when len(obs_var_names) == 1
    obs_matrix = np.squeeze(obs_matrix)
    error_matrix = np.squeeze(error_matrix)
    # check if num of dates == num of observations
#    if obs_matrix.shape[0] != len(dates_obs):
#        raise Exception("Number of dates different of number of obs files")
    return obs_matrix, error_matrix


def generate_dates(date_ini, date_end, timestep=1):

    del_t = [date_ini]
    date_time = date_ini
    while date_time < date_end:
        date_time += dt.timedelta(hours=timestep)

        del_t.append(date_time)
    if date_end != del_t[-1]:
        raise Exception(' Wrong date_ini or date_end (or both), '
                        'not compatible (or worng) time_step')

    del_t = np.asarray(del_t)
    return del_t


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
    del_t = generate_dates(date_ini, date_end)

    files = glob.glob(nc_forcing_path + "*.nc")
    files.sort()

    array_nc = []

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

    return lat_idx, lon_idx


def get_dims(return_ncdim=False):

    nc_forcing_path = cfg.nc_forcing_path
    forcing_dim_names = cfg.forcing_dim_names

    example_file = glob.glob(nc_forcing_path + "*.nc")[0]
    example_file = nc.Dataset(example_file)

    lat_name_var = forcing_dim_names["lat_forz_var_name"]
    lon_name_var = forcing_dim_names["lon_forz_var_name"]
    if return_ncdim:
        lon = example_file.variables[lon_name_var]
        lat = example_file.variables[lat_name_var]
        return lat, lon
    n_lats = len(example_file.variables[lat_name_var][:])
    n_lons = len(example_file.variables[lon_name_var][:])
    return n_lats, n_lons


def forcing_check(forcing_df):

    if forcing_df.isnull().values.any():
        return True

    else:
        return False


def expand_grid():

    mask = cfg.nc_maks_path

    n_lats, n_lons = get_dims()
    grid = np.meshgrid(range(n_lats), range(n_lons))
    grid = np.array(grid).reshape(2, n_lats * n_lons).T

    if mask:  # If mask exists, return string if masked
        mask = nc.Dataset(mask)
        mask_value = mask.variables['mask'][:]
        mask.close()
        mask = mask_value.flatten('F')

        grid = grid[mask == 1]
        grid = np.squeeze(grid)

    return grid


def simulation_steps(observations, dates_obs):

    date_ini = cfg.date_ini
    date_end = cfg.date_end
    season_ini_day = cfg.season_ini_day
    season_ini_month = cfg.season_ini_month
    da_algorithm = cfg.da_algorithm

    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")

    del_t = generate_dates(date_ini, date_end)

    obs_idx = np.searchsorted(del_t, dates_obs)

    # Remove observation NaNs from simulations steps
    if observations.ndim == 1:
        obs_values = observations[obs_idx]
        check = ~np.isnan(obs_values)
        obs_idx = obs_idx[check]
    else:
        obs_values = observations[obs_idx, :]
        check = ~np.all(np.isnan(obs_values), axis=1)
        obs_idx = obs_idx[check]

    days = [date.day for date in del_t]
    months = [date.month for date in del_t]
    hours = [date.hour for date in del_t]

    season_ini_cuts = np.argwhere((np.asarray(days) == season_ini_day) &
                                  (np.asarray(months) == season_ini_month) &
                                  (np.asarray(hours) == 0))

    if cfg.load_prev_run:
        assimilation_steps = 0
    else:
        if da_algorithm in ['PBS', 'ES', 'IES', 'IES-MCMC', 'IES-MCMC_AI',
                            'PIES', 'AdaPBS', 'AdaMuPBS']:
            assimilation_steps = season_ini_cuts[:, 0]
        elif (da_algorithm in ['PF', 'EnKF', 'IEnKF']):
            # HACK: I add one to easy the subset of the forcing
            assimilation_steps = obs_idx + 1
        elif (da_algorithm == 'deterministic_OL'):
            assimilation_steps = 0
        else:
            raise Exception("Choose between smoothing or filtering")

    lng_del_t = np.asarray(len(del_t))
    assimilation_steps = np.append(0, assimilation_steps)
    assimilation_steps = np.append(assimilation_steps, lng_del_t)
    assimilation_steps = np.unique(assimilation_steps)

    return {"del_t": del_t,
            "obs_idx": obs_idx,
            "Assimilaiton_steps": assimilation_steps}


def run_model_openloop(lat_idx, lon_idx, main_forcing, filename):

    print("No observations in: " + str(lat_idx) + "," + str(lon_idx))
    # create temporal simulation
    temp_dest = model.model_copy(lat_idx, lon_idx)
    real_forcing = main_forcing.copy()
    model.model_forcing_wrt(real_forcing, temp_dest, step=0)
    if cfg.numerical_model in ['FSM2']:
        model.model_run(temp_dest)
        state = model.model_read_output(temp_dest, read_dump=False)
    elif cfg.numerical_model in ['dIm', 'snow17']:
        state = model.model_run(real_forcing)[0]
    else:
        Exception("Numerical model not implemented")
    state.columns = list(model.model_columns)

    io_write(filename, state)
    # Clean tmp directory
    try:
        shutil.rmtree(os.path.split(temp_dest)[0], ignore_errors=True)
    except TypeError:
        pass
