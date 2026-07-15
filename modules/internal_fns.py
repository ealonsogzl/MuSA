#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal functions to read and tidy the forcing, as well as launch the real
assimilation functions.

Author: Esteban Alonso González - alonsoe@ipe.csic.es

This file has been modified with new functions and bug fixes by:
    - Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
"""

from multiprocessing import TimeoutError
import glob, os, sys, shutil, pickle, blosc, warnings, re, gc, subprocess, time
import datetime as dt
import netCDF4 as nc
import numpy as np
import multiprocessing as mp
import pandas as pd
import xarray as xr
import config as cfg
if cfg.numerical_model == 'FSM2':
    import modules.fsm_tools as model
elif cfg.numerical_model == 'dIm':
    import modules.dIm_tools as model
elif cfg.numerical_model == 'snow17':
    import modules.snow17_tools as model
else:
    raise Exception('Model not implemented')
import pdcast as pdc
if cfg.MPI:
    from mpi4py.futures import MPIPoolExecutor



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
        warnings.warn("timeout is ignored with MPI")

    if cfg.parallelization == "HPC.array" and cfg.numerical_model == "FSM2":
        fsm_filename = os.path.join(cfg.fsm_src_path, "FSM2")
        if os.path.isfile(fsm_filename):
            warnings.warn("FSM binary exists, reusing compile options")

    if cfg.write_stat_full and cfg.restart_run:
        raise Exception(
            "write_stat_full and restart_run cannot be activated \
                simultaneously. To be implemented.")

    if cfg.write_stat_daily and cfg.restart_run:
        raise Exception(
            "write_stat_daily and restart_run cannot be activated \
                simultaneously. To be implemented.")
    if cfg.da_algorithm not in ["ES", "IES"] and cfg.implementation ==\
            'Spatial_propagation':
        raise Exception("Spatial_propagation needs ES/IES methods")

    if cfg.spatial_in_mem and cfg.parallelization == "HPC.array":
        raise Exception("spatial_in_mem not compatible with HPC.array")


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


def io_write(filename_or_obj, obj=None, codec="lz4", clevel=3, in_mem=False):
    """
    Serializa y comprime un objeto con pickle+blosc.
    Si in_mem=True, devuelve un buffer de bytes.
    Si in_mem=False, escribe a un archivo.
    """
    pickled = pickle.dumps(filename_or_obj if in_mem else obj,
                           protocol=pickle.HIGHEST_PROTOCOL)

    compressed = blosc.compress(pickled, cname=codec, clevel=clevel)

    if in_mem:
        return compressed
    else:
        with open(filename_or_obj, "wb") as f:
            f.write(compressed)


def io_read(source, in_mem=False):
    """
    Lee un objeto serializado con io_write.
    Si in_mem=True, 'source' debe ser un buffer de bytes.
    Si in_mem=False, 'source' es el filename.
    """
    if in_mem:
        compressed = source
    else:
        with open(source, "rb") as f:
            compressed = f.read()

    decompressed = blosc.decompress(compressed)
    return pickle.loads(decompressed)


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
    """Split a list into chunks of size `size`."""
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def pool_wrap(func, inputs, nprocess, timeout=None):
    """
    Ejecuta `func` en paralelo con starmap.
    `inputs` = lista de tuplas.
    Devuelve la lista de resultados de func().
    """
    if cfg.MPI:
        with MPIPoolExecutor() as pool:
            return pool.starmap(func, inputs, timeout=timeout)

    else:
        with mp.Pool(processes=nprocess) as pool:
            async_result = pool.starmap_async(func, inputs)
            try:
                return async_result.get(timeout=timeout)

            except TimeoutError:
                pool.terminate()
                pool.join()
                gc.collect()
                raise

            except Exception:
                pool.terminate()
                pool.join()
                gc.collect()
                raise


def safe_pool(func, inputs, nprocess, in_mem=False):
    """
    Ejecuta func en paralelo por chunks.
    Si in_mem=True → devuelve un diccionario con los resultados acumulados.
    Si in_mem=False → no devuelve nada (comportamiento original).

    inputs debe ser una lista de listas:
        inputs = [arg1_list, arg2_list, ...]
    """
    # Normaliza inputs para evitar problemas con iteradores
    inputs = [list(arg) for arg in inputs]

    cells_per_process = cfg.cells_per_process or 1
    timeout = cfg.timeout

    ncellsmax = cells_per_process * nprocess

    # chunk por cada argumento independiente
    inputs_chunks = [chunker(arg, ncellsmax) for arg in inputs]
    nchunks = len(inputs_chunks[0])

    # Diccionario acumulado en memoria si se solicita
    results_dict = {} if in_mem else None

    for chunk_id in range(nchunks):

        # Extrae el chunk de cada argumento
        chunk_args = [arg_chunks[chunk_id] for arg_chunks in inputs_chunks]

        # Empaqueta en lista de tuplas para starmap
        chunk_input = list(zip(*chunk_args))

        # Retry loop en caso de freeze
        while True:
            try:
                # Ejecuta el pool y recoge resultados
                results = pool_wrap(func, chunk_input,
                                    nprocess, timeout=timeout)

                # Si hay que acumular resultados en memoria
                if in_mem:
                    for r in results:
                        if isinstance(r, dict):
                            results_dict.update(r)
                        else:
                            # si no es dict, generamos clave automáticamente
                            results_dict[len(results_dict)] = r

                break  # chunk completado → siguiente chunk

            except TimeoutError:
                print("Pool frozen. Restarting chunk", chunk_id)
                time.sleep(2)
                continue

    return results_dict


def get_dates_obs():

    dates_obs = cfg.dates_obs

    if type(dates_obs) is list:

        dates_obs.sort()
        dates_obs = np.asarray([dt.datetime.strptime(date, "%Y-%m-%d %H:%M")
                                for date in dates_obs])
    elif type(dates_obs) is str:

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
    #       raise Exception("Number of dates different of number of obs files")

    # add lowest value possible to avoid numerical issues if for some reason
    # r_cov == 0
    error_matrix = error_matrix + np.finfo(type(error_matrix[0])).eps
    return obs_matrix, error_matrix


def generate_dates(date_ini, date_end, timestep=cfg.dt):
    """
    Generate a list of dates starting from date_ini to date_end with a given
    timestep in seconds.

    Args:
        date_ini (datetime): Start date and time.
        date_end (datetime): End date and time.
        timestep (int): Timestep in seconds.

    Returns:
        numpy.ndarray: Array of datetime objects.

    Raises:
        Exception: If the final date in the array does not match date_end.
    """
    if not isinstance(timestep, (int, float)) or timestep <= 0:
        raise ValueError("timestep must be a positive number in seconds.")

    del_t = [date_ini]
    date_time = date_ini
    time_delta = dt.timedelta(seconds=timestep)

    while date_time < date_end:
        date_time += time_delta
        if date_time <= date_end:
            del_t.append(date_time)

    if date_end != del_t[-1]:
        raise Exception("Wrong date_ini or date_end (or both), \
                        not compatible with the given timestep.")

    return np.asarray(del_t)

def check_forcings_timerange(
        date_ini:str="2018-09-01 00:00",
        date_end:str="2020-08-30 23:00",
        forcing_dir:str=os.getcwd(),
        verbose:bool=False
    ) -> str|list[str]:
    '''
    Function that checks if the forcing zarr already exists given the specified date_ini and date_end. 
    If it does, it returns the file, otherwise it returns None.

    Only if no zarr store is found within the specified date_range, it will return None. 
    This None will be then used to trigger the creation of a new zarr store containing the forcings. 
    '''
    files=glob.glob(os.path.join(forcing_dir, "*"))

    if not files:
        if verbose:
            print(f"No forcing files found in {forcing_dir}.", file=sys.stderr)
        return None
    
    elif any(re.search(r".zarr", f) for f in files):
        if verbose:
            print(f"Found forcing zarr files in {forcing_dir}. Checking for the specified date range: {date_ini} to {date_end}", file=sys.stderr)

        if date_ini is None or date_end is None:
            store_search=re.compile(rf"forcings.zarr")
        else:
            date_ini_str=pd.Timestamp(date_ini).strftime('%Y%m%d')
            date_end_str=pd.Timestamp(date_end).strftime('%Y%m%d')
            
            store_search=re.compile(rf"forcings(.+?){date_ini_str}(.+?){date_end_str}.zarr")
        
        forcing_file=next((f for f in glob.glob(os.path.join(forcing_dir, "*")) if store_search.search(f)), None)
        if forcing_file is not None:
            return forcing_file
        else:
            if verbose:
                print(f"No forcing zarr files found in {forcing_dir} for the specified date range!", file=sys.stderr)
            return None
        
    else:
        if verbose:
            print(f"No forcing zarr files found in {forcing_dir}", file=sys.stderr)

        files=sorted(glob.glob(os.path.join(forcing_dir, "*.nc")))
        if date_ini is None or date_end is None:
            return files
        else:
            date_ini_str=pd.Timestamp(date_ini)
            date_end_str=pd.Timestamp(date_end)

            date_range=pd.date_range(start=date_ini_str, end=date_end_str, freq='D')

            if len(files) == len(date_range):
                return files
            else:
                raise ValueError("Number of nc-files does not match the specified date_range. Please remove the files outside this date range.")

def check_vars_in_forcings(
    nc_forcing_path:str = "test/y034x201/FORCINGS",
    date_ini: str = "2015-09-01 00:00",
    date_end: str = "2024-08-31 21:00",
    vars_to_check:list[str] = ["vegh", "VAI", "alb0", "asmn", "asmx", "eta0",
        "hfsn", "kfix", "rcld", "rfix", "rgr0", "rhof", "rhow", "rmlt", "Salb", "snda",
        "Talb", "tcld", "tmlt", "trho", "Wirr", "z0sn", "fcly", "fsnd", "gsat", "z0sf",
        "acn0", "acns", "avg0", "avgs", "cvai", "hbas", "gsnf", "kext", "leaf", "svai",
        "tunl", "wcan"]
    ) -> tuple[list[str], list[str]]:
    '''
    Function that checks if the specified variables are present in the forcing files within the specified date range.
    It returns a tuple containing two lists: the first list contains the variables that are present, and the second list contains the variables that are missing.
    '''
    #---extract the forcing files within the specified date range---
    files=check_forcings_timerange(
        date_ini=date_ini,
        date_end=date_end,
        forcing_dir=nc_forcing_path,
        verbose=False
    )

    #---extract the data variables from the forcing files---
    if files is None:
        raise ValueError(f"No forcing files found in {nc_forcing_path} for the specified date range: {date_ini} to {date_end}.")

    if isinstance(files, list):
        with xr.open_dataset(files[0], chunks={}) as ds:
            dvars=list(ds.data_vars)
    else:
        with xr.open_zarr(files, consolidated=True) as ds:
            dvars=list(ds.data_vars)

    non_present=[v for v in vars_to_check if v not in dvars]
    present=[v for v in vars_to_check if v in dvars]

    return non_present, present

def load_array_forcing(
    date_ini: str = "2015-09-01 00:00",
    date_end: str = "2024-08-31 21:00",
    nc_forcing_path:str = "test/y034x201/FORCINGS",
    var_names: str|list[str] = ["P", "T", "Ua", "LWd", "SWd"],
    lat_idx:int = 0,
    lon_idx:int = 0
    ) -> dict[str, np.ndarray]:
    '''
    New function to load the forcing data from the specified directory and date range. 

    author: Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
    '''
    #---extract the forcing files within the specified date range---
    files=check_forcings_timerange(
        date_ini=date_ini,
        date_end=date_end,
        forcing_dir=nc_forcing_path,
        verbose=False
    )

    #---extract the data variables from the forcing files---
    dict_array_nc = {}
    if not isinstance(files, list):
        with xr.open_zarr(files, consolidated=True) as ds_forcings:
            ds_forcing=ds_forcings.sel(date=slice(date_ini, date_end))[var_names]
            ds_forcing=ds_forcing.stack(datetime=("date", "time"))
            
            ds_forcing=ds_forcing.isel(lat=lat_idx, lon=lon_idx).compute()
            for nc_var in var_names:
                dict_array_nc[nc_var] = np.asarray(ds_forcing[nc_var].values)
    else:
        for nc_var in var_names:
            array_nc = []
            for ncfile in files:
                data_temp = nc.Dataset(ncfile)
                array_temp = data_temp.variables[nc_var][:, lat_idx, lon_idx]
                array_temp = np.ma.getdata(array_temp)
                array_nc.extend(array_temp)
                data_temp.close()
        dict_array_nc[nc_var] = np.array(array_nc)

    # --- check if the length of the forcing data matches the expected length based on the date range and timestep ---
    #change date ini and end and generate del_t
    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
    del_t = generate_dates(date_ini, date_end)

    if len(del_t) != len(dict_array_nc[nc_var]):
            raise Exception("date_end - date_ini longuer than forcing")

    return dict_array_nc

def nc_array_forcing(nc_forcing_path, lat_idx, lon_idx, nc_var_name,
                     date_ini, date_end):
    """
    Extract an array of forcing timesteps from an nc
    Parameters
    ----------
    nc_forcing_path : string
           Path of the the forcings: either multiple netCDF files or a zarr store.
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
    #change date ini and end and generate del_t
    date_ini = dt.datetime.strptime(date_ini, "%Y-%m-%d %H:%M")
    date_end = dt.datetime.strptime(date_end, "%Y-%m-%d %H:%M")
    del_t = generate_dates(date_ini, date_end)

    #list up the files
    files=glob.glob(os.path.join(nc_forcing_path, "*.nc"))
    files.sort()

    if not files:
        #find the zarr store -> expects only one zarr store in the directory!!
        store=next(iter(glob.glob(os.path.join(nc_forcing_path, "*.zarr"))), None)
        if store is None:
            raise FileNotFoundError("No netCDF or zarr files found in the specified directory.")
        #open the store and extract the values
        ds_forcings=xr.open_zarr(store)
        ds_forcing=ds_forcings.sel(date=slice(date_ini, date_end))[nc_var_name]
        ds_forcing=ds_forcing.stack(datetime=("date", "time"))

        array_nc=np.asarray(ds_forcing.isel(lat=lat_idx, lon=lon_idx).values)

        ds_forcings.close()
        del ds_forcings, ds_forcing
    else:
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
    ''' 
    Function that determines the dimensions of the forcing data. 
    '''

    nc_forcing_path = cfg.nc_forcing_path
    forcing_dim_names = cfg.forcing_dim_names

    #search for the example file, if no netCDF file is found, search for a zarr store
    example_file = glob.glob(os.path.join(nc_forcing_path, "*.nc"))
    if len(example_file) == 0:
        try:
            example_file=next(iter(glob.glob(os.path.join(nc_forcing_path, "*.zarr"))))
            example_file=xr.open_zarr(example_file)
        except:
            raise Exception('Forcing files not found')
    else:
        example_file = example_file[0]
        example_file = nc.Dataset(example_file)

    #get the dimensions of the forcing data, either as netCDF dimensions or as lengths of the lat/lon arrays
    lat_name_var = forcing_dim_names["lat_forz_var_name"]
    lon_name_var = forcing_dim_names["lon_forz_var_name"]
    
    try:
        if return_ncdim:
            lon = example_file.variables[lon_name_var]
            lat = example_file.variables[lat_name_var]
            return lat, lon
        else:
            n_lats = len(example_file.variables[lat_name_var][:])
            n_lons = len(example_file.variables[lon_name_var][:])
    except:
        try:
            if return_ncdim:
                lon = example_file[lon_name_var]
                lat = example_file[lat_name_var]
                return lat, lon
            else:
                n_lats = len(example_file[lat_name_var])
                n_lons = len(example_file[lon_name_var])
        except:
            raise Exception('Could not determine dimensions of forcing data')
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

        if np.ndim(grid) == 1: #HACK: if only one cell in the AOI, reshape to 2D array
            grid = grid.reshape(1, -1)  

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
                            'PIES', 'ProPBS', 'AdaPBS']:
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
            "Assimilation_steps": assimilation_steps}


def run_model_openloop(lat_idx, lon_idx, main_forcing, filename):

    if cfg.da_algorithm != 'deterministic_OL':
        print("No observations in: " + str(lat_idx) + "," + str(lon_idx))
    profile = getattr(cfg, "profile_timing", False)

    # create temporal simulation
    start_time = time.perf_counter()
    temp_dest = model.model_copy(lat_idx, lon_idx)
    copy_time = time.perf_counter() - start_time

    real_forcing = main_forcing.copy()
    start_time = time.perf_counter()
    model.model_forcing_wrt(real_forcing, temp_dest, step=0)
    forcing_time = time.perf_counter() - start_time

    if cfg.numerical_model in ['FSM2']:
        start_time = time.perf_counter()
        model.model_run(temp_dest)
        run_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        state = model.model_read_output(temp_dest, read_dump=False)
        read_time = time.perf_counter() - start_time
    elif cfg.numerical_model in ['dIm', 'snow17']:
        start_time = time.perf_counter()
        state = model.model_run(real_forcing)[0]
        run_time = time.perf_counter() - start_time
        read_time = 0.0
    else:
        Exception("Numerical model not implemented")
    state.columns = list(model.model_columns)

    start_time = time.perf_counter()
    io_write(filename, state)
    write_time = time.perf_counter() - start_time

    if profile:
        total_time = copy_time + forcing_time + run_time + read_time + write_time
        print(
            "[FSM timing] cell ({}, {}) total: {:.3f} s | copy: {:.3f} s | "
            "prepare forcing: {:.3f} s | run: {:.3f} s | read: {:.3f} s | "
            "write: {:.3f} s".format(
                lat_idx, lon_idx, total_time, copy_time, forcing_time,
                run_time, read_time, write_time))
    # Clean tmp directory
    try:
        shutil.rmtree(os.path.split(temp_dest)[0], ignore_errors=True)
    except TypeError:
        pass

def open_loop_simulation(lat_idx, lon_idx) -> None:
    ''' 
    Function to perform the open-loop simulation without the need of having 
    observations. Usefull to test our setup with the downscaled data.

    for more infromation contact:
        - lucas.boeykens@ugent.be
        - lucas.boeykens@kuleueven.be
    '''
    total_start = time.perf_counter()
    pid = os.getpid()

    print(
        f"[CELL START] ({lat_idx}, {lon_idx}) pid={pid}",
        flush=True
    )


    #real restart information
    real_time_restart = cfg.real_time_restart_OL

    if real_time_restart:
        name_restart = "init_" + str(lat_idx) +\
            "_" + str(lon_idx) + ".pkl.blp"
        name_restart = os.path.join(cfg.real_time_restart_path_OL, name_restart)

    if cfg.load_prev_run_OL:
        filename = ("Reconstructed_cell_OL_" + str(lat_idx)
                    + "_" + str(lon_idx) + ".pkl.blp")
    else:
        filename = ("cell_" + str(lat_idx) + "_" + str(lon_idx) + ".pkl.blp")

    filename = os.path.join(cfg.output_path, filename)


    # Check if file allready exist if is a restart run
    if (cfg.restart_run and os.path.exists(filename)):
        print(
            f"[CELL SKIPPED] ({lat_idx}, {lon_idx})",
            flush=True
        )
        return None

    #get the forcings
    start = time.perf_counter()
    main_forcing = model.forcing_table(lat_idx, lon_idx)
    forcing_load_time = time.perf_counter() - start

    start = time.perf_counter()
    invalid_forcing = forcing_check(main_forcing)
    forcing_check_time = time.perf_counter() - start

    if invalid_forcing:
        print("NA's found in: " + str(lat_idx) + "," + str(lon_idx))
        return None

    #do the OL without any DA
    start = time.perf_counter()
    run_model_openloop(lat_idx, lon_idx, main_forcing, filename)
    model_time = time.perf_counter() - start

    total_time = time.perf_counter() - total_start

    print(
        f"[CELL END] ({lat_idx}, {lon_idx}) | "
        f"total={total_time:.3f}s | "
        f"forcing load={forcing_load_time:.3f}s | "
        f"forcing check={forcing_check_time:.3f}s | "
        f"model={model_time:.3f}s",
        flush=True
    )

    return None