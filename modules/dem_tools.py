#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script containing functions to read in the DEM file corresponding to the tile of 
the meteorological forcing data, and to prepare the DEM for the MuSA model.

The functions expects the forcings to be devided in tiles using the naming convention "y{ty:03d}x{tx:03d}" in the folder name, 
where ty and tx are the tile coordinates.

# author: Lucas Boeykens - Lucas.boeykens@ugent.be, lucas.boeykens@kuleuven.be
"""

#---modules---
import os, sys, re, glob
import xarray as xr
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.OperationsXrDatasets import transpose_dataset, saveXrtoNetCDF

#---functions---
def RegridDEMtoForcings(
        rootdir_dem:str="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/auxdata/dem/cop90/100m",
        datadir_forcings:str=None, 
        savedir:str=os.getcwd(),
        overwrite:bool=False
    ) -> tuple[str, int]:
    ''' 
    Function that searches for the DEM file corresponding to the tile of the forcings,
    upscales the DEM to the same resolution as the forcing data,
    makes sure both datasets have the same grid and saves the regridded DEM to a netcdf file.
    '''
    #open the file with forcing data
    ds_forcings=next((iter(glob.glob(os.path.join(datadir_forcings, "*")))), None)

    if not ds_forcings:
        raise ValueError(f"No forcing file found in {datadir_forcings}")
    
    if ds_forcings.endswith(".zarr"):
        ds_forcings=xr.open_zarr(ds_forcings,consolidated=True)
    elif ds_forcings.endswith(".nc"):
        ds_forcings=xr.open_dataset(ds_forcings)
    else:
        raise ValueError(f"Unsupported file format for forcings: {ds_forcings}")

    #extract and open the dem-file
    tile_str=re.search(r"y([0-9]{3})x([0-9]{3})", datadir_forcings)
    ty=int(tile_str.group(1))
    tx=int(tile_str.group(2))

    demFile=next(iter(glob.glob(os.path.join(rootdir_dem, f"y{ty:03d}x{tx:03d}*nc"))))
    ds_dem=transpose_dataset(xr.open_dataset(demFile))

    #upscale and make sure same grid -> NOTE: the original res of the DEM is 100m!!
    upscale_factor=ds_dem.sizes["lat"]//ds_forcings.sizes["lat"]

    ds_dem=ds_dem.coarsen(lat=upscale_factor, lon=upscale_factor, side="left").mean()
    ds_dem["lat"]=ds_forcings["lat"] #HARD CODED: works for now, but not always the case
    ds_dem["lon"]=ds_forcings["lon"] #HARD CODED: works for now, but not always the case

    #get the DEM variable name and resolution
    dem_var=next(d for d in ds_dem.data_vars if "dem" in d.lower())
    dem_res=upscale_factor*100

    #save the regridded DEM to a netcdf file -> only if not existent
    if not os.path.exists(os.path.join(savedir, "dem_regridded.nc")):
        saveXrtoNetCDF(ds=ds_dem, 
                    savedir=savedir, 
                    filename="dem_regridded.nc"
                    )
        print(f"DEM regridded to forcings successfully. DEM variable name: {dem_var}, DEM resolution: {dem_res} m.", file=sys.stderr)

    elif os.path.exists(os.path.join(savedir, "dem_regridded.nc")) and overwrite:
        saveXrtoNetCDF(ds=ds_dem, 
                    savedir=savedir, 
                    filename="dem_regridded.nc"
                    )
        print(f"DEM regridded to forcings successfully. DEM variable name: {dem_var}, DEM resolution: {dem_res} m.", file=sys.stderr)
    else:
        print(f"DEM regridded to forcings already exists. DEM variable name: {dem_var}, DEM resolution: {dem_res} m.", file=sys.stderr)
        
    return dem_var, dem_res