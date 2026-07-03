import os, sys, glob, re, argparse, importlib
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
project_root=os.getcwd()
sys.path.append(project_root)
import modules.prepareForcingsZarr as prepForcings
from utils.OperationsXrDatasets import saveXrtoNetCDF
importlib.reload(prepForcings)
from modules.dem_tools import RegridDEMtoForcings
import config_template as cfg #HARD CODED: this is the config fil that is adjusted for the specific experiment

#---custom functions---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Regrid DEM to Forcings")
    parser.add_argument("--tx", 
                        type=int,
                        default=201,
                        help="Tile x-coordinate (default: 201)")
    parser.add_argument("--ty",
                        type=int,
                        default=35,
                        help="Tile y-coordinate (default: 35)")
    parser.add_argument("--project_root",
                        type=str,
                        default=os.getcwd(),
                        help="Root directory of the project")
    parser.add_argument("--implementation",
                        type=str,
                        default="open_loop",
                        help="Implementation type (default: open_loop)")
    
    parser.add_argument("--date_ini",
                        type=str,
                        default="2018-09-01 00:00",
                        help="Initial date for the simulation (default: 2018-09-01 00:00)")

    parser.add_argument("--date_end",
                        type=str,
                        default="2020-08-30 23:00",
                        help="End date for the simulation (default: 2020-08-30 21:00)")

    parser.add_argument("--snow_model",
                        type=str,
                        default="FSM2",
                        help="Snow model to use (default: FSM2)")
    parser.add_argument("--rootdirMuSAruns",
                        type=str,
                        default=os.getcwd(),
                        help="Root directory for MuSA runs (default: current working directory)"
                        )
    parser.add_argument("--model_only_sites",
                        type=bool,
                        default=False,
                        help="Flag to indicate if only model sites should be considered (default: False)"
                        )

    args = parser.parse_args()
    
    return args

def check_forcings_timerange(
        date_ini:str="2018-09-01 00:00",
        date_end:str="2020-08-30 23:00",
        forcing_dir:str=os.getcwd()
    ) -> bool:
    '''
    Function that checks if the forcing zarr already exists given the specified date_ini and date_end. 
    If it does, it returns True, otherwise it returns False.
    '''
    date_ini_str=pd.Timestamp(date_ini).strftime('%Y%m%d')
    date_end_str=pd.Timestamp(date_end).strftime('%Y%m%d')


    store_search=re.compile(rf"forcings.zarr") if not date_ini and not date_end \
        else re.compile(rf"forcings(.+?){date_ini_str}(.+?){date_end_str}.zarr")
    
    if any(re.search(store_search, f) for f in glob.glob(os.path.join(forcing_dir, "*"))):
        return True
    else:
        return False

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

def save_config_module(
        savedir:str=None
        ) -> str:
    ''' 
    Function that saves a config file load in as a python module.
    It saves the config file as a python module with the same name as the original config file, 
    but with the suffix "_modified" added to the name.
    '''
    #adjust the output path to be a Path object
    out_path = Path(os.path.join(savedir, "config_run.py")) if savedir else Path("config_run.py")

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
        model_only_sites:bool=False
    ) -> None:
    ''' 
    Function that adjusts certain variables in the custom_config.py file to point to the correct directories for the current run.
    '''
    def _check_date_ini_end(
        date_ini:str,
        date_end:str,
        ds_forcings:xr.Dataset,
        time_dim:str
        ) -> tuple[str,str]:
        ''' 
        Helper function that checks if the date_ini and date_end are within the range of the forcing dataset.
        If not, it adjusts the date_ini and date_end to be within the range of the forcing dataset.
        '''
        #get the original starting and end hour of the date_ini and date_end
        h_begin=int(date_ini.split(" ")[-1].split(":")[0])
        h_end=int(date_end.split(" ")[-1].split(":")[0])

        #adjust based on the forcing dataset
        h_begin_adjusted=int(ds_forcings[time_dim].values[0].item())
        h_end_adjusted=int(ds_forcings[time_dim].values[-1].item())

        date_ini_adjusted=date_ini.replace(f"{h_begin:02d}:00", f"{h_begin_adjusted:02d}:00")
        date_end_adjusted=date_end.replace(f"{h_end:02d}:00", f"{h_end_adjusted:02d}:00")
        
        # h_begin_adjusted=int(ds_forcings[time_dim][np.argmin(np.abs(ds_forcings[time_dim].values - h_begin))].item())
        # h_end_adjusted=int(ds_forcings[time_dim][np.argmin(np.abs(ds_forcings[time_dim].values - h_end))].item())

        return date_ini_adjusted, date_end_adjusted

    def _CreateMaskMsitesTile(
            store_measurements:str="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/measurements/insitu/Alps_dataset_SD.nc",
            tx:int=200, 
            ty:int=35) -> None:
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

    #HARD CODED directories based on the rootdirRun
    forcing_dir=os.path.join(rootdirRun,"FORCINGS")
    dem_dir=os.path.join(rootdirRun,"DEM")

    #---open the forcing data -> extract lat, lon and time dimensions---
    forcing_file=next(iter(glob.glob(os.path.join(forcing_dir, "*"))))
    if not forcing_file:
        raise ValueError(f"No forcing file found in {forcing_dir}")

    if forcing_file.endswith(".zarr"):
        ds_forcings=xr.open_zarr(forcing_file,consolidated=True)
    elif forcing_file.endswith(".nc"):
        ds_forcings=xr.open_dataset(forcing_file)
    else:
        raise ValueError(f"Unsupported file format for forcings: {forcing_file}")

    lat_dim=next((d for d in ds_forcings.dims if re.fullmatch(r"lat|latitude|northing", d, re.IGNORECASE)), 
                 None
                 )
    lon_dim=next((d for d in ds_forcings.dims if re.fullmatch(r"lon|longitude|easting", d, re.IGNORECASE)), 
                 None
                 )
    time_dim=next((d for d in ds_forcings.dims if re.fullmatch(r"time|t", d, re.IGNORECASE)), 
                  None
                  )

    if any(dim is None for dim in [lat_dim, lon_dim, time_dim]):
        raise ValueError("Could not find latitude, longitude, and time dimensions in the forcing dataset.")

    cfg.forcing_dim_names["lat_forz_var_name"]=lat_dim
    cfg.forcing_dim_names["lon_forz_var_name"]=lon_dim
    cfg.forcing_dim_names["time_forz_var_name"]=time_dim


    #---adjust the implementation type in the config file--
    cfg.implementation=implementation
    if cfg.implementation=="open_loop":
        cfg.da_algorithm="deterministic_OL" #ADDED: make it deterministic OL to not priunt out unnecessary info in ifns.run_model_openloop

    results_folder=os.path.join(rootdirRun, "RESULTS_OL/") if cfg.implementation=="open_loop" \
                        else os.path.join(rootdirRun, f"RESULTS_{cfg.da_algorithm}/")
    intermediate_folder=os.path.join(rootdirRun, "INTERMEDIATE_OL/") if cfg.implementation=="open_loop" \
                        else os.path.join(rootdirRun, f"INTERMEDIATE_{cfg.da_algorithm}/")

    #---paths and directories---
    cfg.nc_obs_path=os.path.join(rootdirRun,"Obs/")
    cfg.nc_forcing_path = forcing_dir
    # cfg.nc_maks_path = mask_dir #TODO: fix in general the name of nc_maks_path to nc_mask_path in the config file and in the code
    cfg.dem_path = os.path.join(dem_dir,"dem_regridded.nc")
    cfg.intermediate_path = intermediate_folder
    cfg.save_ensemble_path = os.path.join(rootdirRun, f"ENSEMBLES_{cfg.da_algorithm}/")
    cfg.output_path = results_folder
    cfg.spatial_propagation_storage_path = os.path.join(rootdirRun, f"SPATIAL_PROP_{cfg.da_algorithm}/")
    cfg.real_time_restart_path = os.path.join(rootdirRun, f"REAL_TIME_RESTART_{cfg.da_algorithm}/")
    cfg.tmp_path = None


    #--adjust the timestep in the config file--
    step_size_forcings=24//ds_forcings.sizes[time_dim]

    cfg.dt=cfg.dt*step_size_forcings

    #---adjust the number of processes in the config file---
    cfg.nprocess=max(nprocess,nprocess_min)

    #--adjust the date_ini and date_end in the config file--
    date_ini, date_end = _check_date_ini_end(
        date_ini=date_ini,
        date_end=date_end,
        ds_forcings=ds_forcings,
        time_dim=time_dim
    )
    cfg.date_ini=date_ini
    cfg.date_end=date_end

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

    #--save the adjusted config file to the rootdirRun--
    out_path=save_config_module(rootdirRun)

    return out_path

def main():
    #load arguments
    args=parse_arguments()

    #create the directories for the run of MuSA for the specific tile (tx,ty)
    rootdirRun, forcing_dir, dem_dir=CreateDirectoriesMuSArunTile(
                                                            tx=args.tx,
                                                            ty=args.ty,
                                                            rootdirMuSAruns=args.rootdirMuSAruns
                                                            )

    #generate the forcings zarr file for the specific tile (tx,ty)
    check_forcings_store=check_forcings_timerange(
            date_ini=args.date_ini,
            date_end=args.date_end,
            forcing_dir=forcing_dir
        )

    if not check_forcings_store:
        prepForcings.CreateZarrTransformedForcings(
            tx=args.tx,
            ty=args.ty,
            date_ini=args.date_ini,
            date_end=args.date_end,
            savedir=forcing_dir,
            filename="forcings.zarr"
            )
        print("Forcings zarr file created successfully.", file=sys.stderr)


    #generate the DEM for the specific tile (tx,ty)
    dem_var, res_dem=RegridDEMtoForcings(
            datadir_forcings=forcing_dir, 
            savedir=dem_dir
        )
    print(f"DEM regridded to forcings successfully. DEM variable name: {dem_var}, DEM resolution: {res_dem} m.", file=sys.stderr)

    #adjust the config-file
    out_path=adjust_config_file(
            snowmodel=args.snow_model,
            rootdirRun=rootdirRun,
            dem_varname=dem_var,
            dem_res=res_dem,
            date_ini=args.date_ini,
            date_end=args.date_end,
            implementation=args.implementation,
            model_only_sites=args.model_only_sites
        )
    print("Preprocessing complete!", file=sys.stderr)

    return out_path

if __name__ == "__main__":
    out_path=main()
    print(out_path)