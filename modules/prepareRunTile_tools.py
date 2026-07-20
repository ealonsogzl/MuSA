''' 
TODO: update this file with the relevant information 

author: Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
'''

#---modules---
import os, sys, glob, re
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.OperationsXrDatasets import saveXrtoNetCDF
import config_template as cfg #HARD CODED: this is the config fil that is adjusted for the specific experiment
import modules.internal_fns as ifns

#---functions---
def _save_config_module(
        savedir:str=None
        ) -> str:
    ''' 
    Function that saves a config file load in as a python module.
    It saves the config file as a python module with the same name as the original config file, 
    but with the suffix "_modified" added to the name.
    '''
    #adjust the output path to be a Path object -> tmp file with random number to avoid overwriting existing files
    savedir=Path(savedir) if savedir else Path(os.getcwd())
    while True:
        random_number=np.random.randint(1e7)
        fileToSave=f"tmp_config_{random_number}.py"

        out_path=Path(os.path.join(savedir, fileToSave))
        if not os.path.exists(out_path):
            break


    #top lines of the config file
    lines = [
        "# Adjusted config file for the run\n",
        "\n",
    ]

    #add the variables from the config file to the lines list
    for name, value in sorted(vars(cfg).items()):
        # skip private/module/system attributes
        if name.startswith("_"):
            continue

        # skip imported modules/functions/classes
        if callable(value):
            continue
        if getattr(value, "__class__", None).__name__ == "module":
            continue

        lines.append(f"{name} = {repr(value)}\n")

    #write the lines to the output file
    out_path.write_text("".join(lines))

    return out_path

def _CreateMaskMsitesTile(
        store_measurements:str="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/measurements/insitu/Alps_dataset_SD.nc",
        tx:int=200, 
        ty:int=35
        ) -> None:
        ''' 
        Helper function that creates a mask for the measurement sites within a specific tile (tx, ty) 
        and saves it as a NetCDF file.
        '''

        dsMeas=xr.open_dataset(store_measurements)

        #---select the tile---
        index=(dsMeas["tx"]==tx) & (dsMeas["ty"]==ty)
        dsMeasTile=dsMeas.where(index, drop=True)

        #---create the mask for the tile---
        mask=xr.open_dataset(cfg.dem_path)
        mask=mask.rename_vars({"dem":"mask"})

        mask=mask*np.nan
        coordsMaskSites=mask.sel(lat=dsMeasTile["lat"], 
                lon=dsMeasTile["lon"], method="nearest")

        point_match = (
        (mask["lat"] == coordsMaskSites["lat"]) &
        (mask["lon"] == coordsMaskSites["lon"])
        )

        mask=xr.where(point_match.any("site"), 1, np.nan).to_dataset(name="mask")

        saveXrtoNetCDF(
        ds=mask,
        savedir=cfg.rootdirRun,
        filename="mask_msites_tile.nc"
        )

        cfg.nc_maks_path=os.path.join(cfg.rootdirRun, "mask_msites_tile.nc")

def _check_data_ini_end_nc(
        forcing_files:list[str],
        date_pattern=re.compile(r"\d{8}")
    ):
    ''' 
    Helper function to automatically extract the date_ini and date_end from a list of nc-files!
    Function is used in check_date_ini_end.
    '''
    #open the first forcing file to get the time dimension and its values
    ds_forcing=xr.open_dataset(forcing_files[0])

    #extract begin and end time from the time dimension of the forcing dataset
    time_dim=next((d for d in ds_forcing.dims if re.fullmatch(r"time|t", d, re.IGNORECASE)), 
                    None
                )
    if time_dim is None:
        raise ValueError("Could not find time dimension in the forcing dataset.")
    ini_time=int(ds_forcing[time_dim].values[0].item())
    end_time=int(ds_forcing[time_dim].values[-1].item())

    #extract the date from the forcing file names using regex
    try:
        start_date=date_pattern.search(forcing_files[0]).group()
        end_date=date_pattern.search(forcing_files[-1]).group()

        start_date=pd.Timestamp(start_date).strftime('%Y-%m-%d')
        end_date=pd.Timestamp(end_date).strftime('%Y-%m-%d')
    except AttributeError:
        raise ValueError("Could not find date in the forcing file names. Please make sure the date is in the format YYYYMMDD in the file names.")

    return f"{start_date} {ini_time:02d}:00", f"{end_date} {end_time:02d}:00"

def _check_data_ini_end_zarr(
        forcing_file:str=None,
    ):
    ''' 
    Helper function to automatically extract the date_ini and date_end from a zarr store of forcings!
    Function is used in check_date_ini_end.
    '''
    if not forcing_file.endswith(".zarr"):
        raise ValueError("Forcing file is not in zarr format. Please provide a zarr store for the forcings.")

    #open the file
    ds_forcing=xr.open_zarr(forcing_file, consolidated=True)

    time_dim=next((d for d in ds_forcing.dims if re.fullmatch(r"time|t", d, re.IGNORECASE)), 
                    None
                )
    date_dim=next((d for d in ds_forcing.dims if re.fullmatch(r"date|d", d, re.IGNORECASE)),
                    None
                )
    if time_dim is None or date_dim is None:
        raise ValueError("Dataset needs time and date dimensions to extract the date_ini and date_end from the forcing dataset.")

    #extract start and end date from the forcing dataset
    start_date=ds_forcing.isel({date_dim: 0})[date_dim].item()
    end_date=ds_forcing.isel({date_dim: -1})[date_dim].item()

    start_date=pd.Timestamp(start_date).strftime('%Y-%m-%d')
    end_date=pd.Timestamp(end_date).strftime('%Y-%m-%d')

    #extract the start and end time from the time dimension of the forcing dataset
    ini_time=int(ds_forcing[time_dim].values[0].item())
    end_time=int(ds_forcing[time_dim].values[-1].item())

    return f"{start_date} {ini_time:02d}:00", f"{end_date} {end_time:02d}:00"

def _check_date_ini_end(
    date_ini:str="2018-09-01 00:00",
    date_end:str="2020-08-30 23:00",
    forcing_dir:str=None,
    ) -> tuple[str,str]:
    ''' 
    Helper function that automatically checks if date_ini and date_end
    are correct to run MuSA. It checks if the date_ini and date_end are within the range of the forcing dataset.
    If not, it raises a ValueError and suggests the correct date_ini and date_end.capitalize

    If correct, it adjusts them to the first and last timestep of the forcing zarr dataset/forcing nc files!

    '''
    forcing_files=ifns.check_forcings_timerange(date_ini=date_ini, 
                                            date_end=date_end, 
                                            forcing_dir=forcing_dir
                                            )

    if isinstance(forcing_files, list):
        date_ini, date_end=_check_data_ini_end_nc(forcing_files=forcing_files)
    else:
        date_ini, date_end=_check_data_ini_end_zarr(forcing_file=forcing_files)

    return date_ini, date_end

def _UpdateConfigPaths(
        rootdirRun:str=None, 
        forcing_dir:str=None, 
        dem_dir:str=None, 
        implementation:str=None,
        date_ini:str="2018-09-01 00:00",
        date_end:str="2020-08-30 23:00"
        ) -> None:
    '''
    Function that adjusts certain path variables in the 
    config.py file to point to the correct directories for the current run. given the 
    arguments provided.
    '''
    #check date_ini and date_end
    if date_ini is not None and date_end is not None:
        date_ini_str=pd.Timestamp(date_ini).strftime('%Y%m%d')
        date_end_str=pd.Timestamp(date_end).strftime('%Y%m%d')

    #adjust the implementation type in the config file--
    cfg.implementation=implementation
    if cfg.implementation=="open_loop":
        cfg.da_algorithm="deterministic_OL" #ADDED: make it deterministic OL to not priunt out unnecessary info in ifns.run_model_openloop

    #adjust results and intermediate folders based on the implementation type
    results_folder=os.path.join(rootdirRun, "RESULTS_OL") if cfg.implementation=="open_loop" \
                        else os.path.join(rootdirRun, f"RESULTS_{cfg.da_algorithm}")
    intermediate_folder=os.path.join(rootdirRun, "INTERMEDIATE_OL") if cfg.implementation=="open_loop" \
                        else os.path.join(rootdirRun, f"INTERMEDIATE_{cfg.da_algorithm}")

    if date_ini is not None and date_end is not None:
        results_folder=results_folder + f"_{date_ini_str}_{date_end_str}/"
        intermediate_folder=intermediate_folder + f"_{date_ini_str}_{date_end_str}/"
        ensemble_path=os.path.join(rootdirRun, f"ENSEMBLES_{cfg.da_algorithm}_{date_ini_str}_{date_end_str}/")
        spatial_propagation_path=os.path.join(rootdirRun, f"SPATIAL_PROP_{cfg.da_algorithm}_{date_ini_str}_{date_end_str}/")
        real_time_restart_path=os.path.join(rootdirRun, f"REAL_TIME_RESTART_{cfg.da_algorithm}_{date_ini_str}_{date_end_str}/")
    else:
        results_folder=results_folder + f"/"
        intermediate_folder=intermediate_folder + f"/"
        ensemble_path=os.path.join(rootdirRun, f"ENSEMBLES_{cfg.da_algorithm}/")
        spatial_propagation_path=os.path.join(rootdirRun, f"SPATIAL_PROP_{cfg.da_algorithm}/")
        real_time_restart_path=os.path.join(rootdirRun, f"REAL_TIME_RESTART_{cfg.da_algorithm}/")


    #---paths and directories---
    cfg.nc_obs_path=os.path.join(rootdirRun,"Obs/")
    cfg.nc_forcing_path = forcing_dir
    # cfg.nc_maks_path = mask_dir #TODO: fix in general the name of nc_maks_path to nc_mask_path in the config file and in the code
    cfg.dem_path = os.path.join(dem_dir,"dem_regridded.nc")
    cfg.intermediate_path = intermediate_folder
    cfg.save_ensemble_path = ensemble_path
    cfg.output_path = results_folder
    cfg.spatial_propagation_storage_path = spatial_propagation_path
    cfg.real_time_restart_path = real_time_restart_path

def adjust_config_file(
        snowmodel:str="FSM2",
        rootdirRun:str=os.getcwd(),
        dem_varname:str="dem",
        dem_res:int=500,
        date_ini:str="2018-09-01 00:00",
        date_end:str="2020-08-30 23:00",
        implementation:str="open_loop",
        nprocess:int=len(os.sched_getaffinity(0)),
        nprocess_min:int=8,
        model_only_sites:bool=False,
        remove_output_cells:bool=False
    ) -> None:
    ''' 
    Function that changes the config file based on the input arguments. 
    It generates and saves the adjusted config file, which is later used to run MuSA. 
    '''
    #HARD CODED directories based on the rootdirRun
    forcing_dir=os.path.join(rootdirRun,"FORCINGS")
    dem_dir=os.path.join(rootdirRun,"DEM")

    #---open the forcing data---
    forcing_file=ifns.check_forcings_timerange(date_ini=date_ini, 
                                        date_end=date_end,
                                        forcing_dir=forcing_dir,
                                        verbose=True)
    if not forcing_file:
        raise ValueError(f"No forcing file found in {forcing_dir}")
    if isinstance(forcing_file, list):
        ds_forcings=xr.open_zarr(forcing_file[0],consolidated=True)
    else:
        ds_forcings=xr.open_zarr(forcing_file,consolidated=True)

    #---extract lat, lon and time dimensions -> from the forcings---
    lat_dim=next((d for d in ds_forcings.dims if re.fullmatch(r"lat|latitude|northing", 
                                                            d, 
                                                            re.IGNORECASE)), 
                    None
                )
    lon_dim=next((d for d in ds_forcings.dims if re.fullmatch(r"lon|longitude|easting", 
                                                            d, 
                                                            re.IGNORECASE)), 
                    None
                )
    time_dim=next((d for d in ds_forcings.dims if re.fullmatch(r"time|t", d, re.IGNORECASE)), 
                    None
                )

    if any(dim is None for dim in [lat_dim, lon_dim, time_dim]):
        raise ValueError(("Could not find latitude, longitude, and "
                        "time dimensions in the forcing dataset."))
    cfg.forcing_dim_names["lat_forz_var_name"]=lat_dim
    cfg.forcing_dim_names["lon_forz_var_name"]=lon_dim
    cfg.forcing_dim_names["time_forz_var_name"]=time_dim

    step_size_forcings=24//ds_forcings.sizes[time_dim] #HARD CODED: assuming that the files are daily with 24 hours
    cfg.dt=cfg.dt*step_size_forcings

    #---update the config paths based on the rootdirRun, forcing_dir, dem_dir, and date_ini and date_end---
    _UpdateConfigPaths(rootdirRun=rootdirRun, 
                    forcing_dir=forcing_dir,
                    dem_dir=dem_dir, 
                    implementation=implementation, 
                    date_ini=date_ini, 
                    date_end=date_end)

    #---adjust the number of processes in the config file---
    cfg.nprocess=max(nprocess,nprocess_min)

    #--adjust the date_ini and date_end in the config file--
    cfg.date_ini, cfg.date_end = _check_date_ini_end(
        date_ini=date_ini,
        date_end=date_end,
        forcing_dir=forcing_dir,
    )

    #---Topographical hyperparameters---
    cfg.DEM_res=dem_res             # DEM resolution
    # TPI_size = 25            # TPI window size
    # Sx_dmax = 15             # Sx search distance
    # Sx_angle = 315           # Sx main wind direction angle
    cfg.nc_dem_varname=dem_varname     # Name of the elevation variable in the DEM

    #--adjust the snow model in the config file--
    cfg.numerical_model=snowmodel

    #--save the rootdir of the run---
    cfg.rootdirRun=rootdirRun

    #--create a mask for the sites in the tile if model_only_sites is True--
    if model_only_sites:
        _CreateMaskMsitesTile(
            tx=int(re.search(r"x(\d+)", rootdirRun).group(1)),
            ty=int(re.search(r"y(\d+)", rootdirRun).group(1))
        )
    else:
        cfg.nc_maks_path=None

    #--adjust the remove_output_cells in the config file--
    cfg.remove_output_cells=remove_output_cells

    #--save the adjusted config file to the rootdirRun--
    out_path=_save_config_module(rootdirRun)

    return out_path

def CreateDirectoriesMuSArunTile(
        tx:int=201,
        ty:int=35,
        rootdirMuSAruns:str=os.getcwd(),
        ) -> tuple[str,str,str]:
    ''' 
    Function that creates the necessary directories for the run of MuSA for a specific tile (tx,ty).
    cuyrrently only one tile at the time is supported!

    TODO: make a folder for the date_ini and date_end of the run
    TODO: make a folder for the observations as well
    '''
    #get the str name of the tile
    tile_str=f"y{ty:03d}x{tx:03d}"

    #make the necessary directories for the run of MuSA for the specific tile (tx,ty) -> forcings and DEM currently
    rootdirRun=os.path.join(rootdirMuSAruns,tile_str)

    dem_dir=os.path.join(rootdirRun,"DEM")
    os.makedirs(dem_dir,exist_ok=True)

    forcing_dir=os.path.join(rootdirRun,"FORCINGS")
    os.makedirs(forcing_dir,exist_ok=True)

    return rootdirRun, forcing_dir, dem_dir