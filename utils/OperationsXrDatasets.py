#---imports---
import numpy as np
import xarray as xr
import pandas as pd
import os,re

#---functions---
def flipLatds(
        ds:xr.Dataset=None
    ) -> xr.Dataset:
    '''
    small function to flip a patch if lat is ascending
    '''
    #extract lat coordinate
    latCoord=next((c for c in ds.coords if re.search(r"y|lat",c)),
                  None)

    #flip if lat is ascending
    if latCoord is not None and (ds[latCoord][1]-ds[latCoord][0])>0:
        ylen = ds.sizes[latCoord]
        ds=ds.isel({latCoord:slice(None, None, -1)})
        #assign new lat coordinates after flipping
        lats=ds[latCoord].values
        ds=ds.assign_coords({latCoord:lats})
    
    return ds

def transpose_dataset(
        ds:xr.Dataset=None
    ) -> xr.Dataset:
    ''' 
    Function to transpose individual data variables in a dataset.
    It checks wether the latitude coordintate is the last dimension,
    if not, it will transpose the dataset to have the correct order for python!
    '''
    #get the lat and lon dimension names
    lat_dim=next((d for d in ds.dims if re.search(r"lat|latitude|y", d, re.IGNORECASE)), None)
    lon_dim=next((d for d in ds.dims if re.search(r"lon|longitude|x", d, re.IGNORECASE)), None)


    for k,v in ds.data_vars.items():
        if not any(latlon_coord in v.dims for latlon_coord in [lat_dim, lon_dim]):
            continue
        #check if a transpose is needed
        if [d for d in v.dims if d in [lat_dim, lon_dim]][0]==lon_dim:
            ds[k]=v.transpose(*[s for s in v.dims if s not in [lat_dim, lon_dim]] + \
                              [lat_dim, lon_dim])
    return ds


def saveXrtoNetCDF(ds:xr.Dataset, 
               savedir:str, 
               filename:str
               )-> None:
    ''' 
    Function to save a xarray dataset to a NetCDF file
    '''
    #error handling
    if not filename.endswith(".nc"):
        raise KeyError("The filename must end with .nc")
    #final adjustments to the dataset and save it
    for k, v in ds.data_vars.items():
        if (v.dtype=="float64") & (k not in ["lat", "lon"]):
            ds[k]=v.astype("float32")
        elif v.dtype=="int64":
            ds[k]=v.astype("int32")
    #deflate
    enc_deflate={k: {'zlib': True, 'complevel': 5} \
            for k,v in ds.data_vars.items() \
                if v.dtype != "object"}
    #save
    os.makedirs(savedir,exist_ok=True)
    outfile=os.path.join(savedir, filename)
    if os.path.exists(outfile):
        os.remove(outfile)
    ds.to_netcdf(outfile, encoding=enc_deflate)
    print(f"Saved the file to: {outfile}")

def target_chunk_len(
    obj:xr.DataArray|xr.Dataset,
    target_mb:int=100,
    chunk_along:str="file_number",
    split_latlon:bool=False,
    chunksize_latlon:int=64,
    safety:float=0.9,  # headroom for temporaries
    ) -> int:

    # convert target MB → number of elements
    bytes_target=target_mb*(1024**2)*safety

    #get the dimensions -> lat and lon if applicable and other dims
    latlondims_obj=[d for d in obj.dims if \
                re.search(r"x|y|lat|lon", d.lower())]
    otherdims_obj=[d for d in obj.dims if d!=chunk_along and \
                not re.search(r"x|y|lat|lon", d.lower())]
                
    if split_latlon and len(latlondims_obj)!=2:
        raise ValueError("Cannot split lat/lon chunks if both lat and lon " \
        "dimensions are not found.")

    if isinstance(obj, xr.DataArray):
        # size of one element
        elem_size = obj.dtype.itemsize
        #make distinction between splitting lat/lon or not
        denom = np.prod(
            [obj.sizes[d] if not split_latlon else chunksize_latlon \
                for d in obj.dims if d in latlondims_obj or d in otherdims_obj]
        )*elem_size
        if denom <= 0:
            t=1
        else:
            t=int(bytes_target // denom)
        chunk_len=max(1, min(t, obj.sizes[chunk_along]))
    elif isinstance(obj, xr.Dataset):
        # sum of sizes of all variables
        total_elem_size = 0
        for v in obj.data_vars.values():
            total_elem_size += v.dtype.itemsize * np.prod(
                [v.sizes[d] if not split_latlon else chunksize_latlon \
                    for d in v.dims if d in latlondims_obj or d in otherdims_obj]
            )
        elems_target=bytes_target
        chunk_len=max(1, elems_target // total_elem_size)
    else:
        raise ValueError("obj must be an xarray DataArray or Dataset.")

    return chunk_len

def rechunk_zarr(
            store:str|xr.Dataset=None, 
            dim_to_rechunk:str="file_number", 
            target_mb_per_chunk:int=100,
            chunk_total_ds:bool=False,
            chunk_latlon:bool=False,
            chunksize_latlon:int=64
            ) -> None:
    ''' 
    Function to rechunk a zarr store or xarray dataset along a specific dimension
    to target a specific chunk size in MB.
    If a string is provided, it is assumed to be a path to a zarr store.
    If a dataset is provided, it is used directly.
    The rechunking is done using xarray's chunking capabilities.
    '''
    #---open the store if a string is provided---
    if isinstance(store, str):
        ds = xr.open_zarr(store, consolidated=True)
    else:
        ds = store

    #---get the latlon dimensions---
    latlondims_ds=[d for d in ds.dims if \
                re.search(r"x|y|lat|lon", d.lower())]

    #---calculate the chunk sizes---
    obj=ds if chunk_total_ds else ds[next(dvar for dvar in ds.data_vars if \
                                        np.issubdtype(ds[dvar].dtype, np.floating))]

    dict_rechunk={}
    for var,data in ds.data_vars.items():
        #salient variable check
        if var=="spatial_ref":
            continue
        #compute the chunk length along the dimension to rechunk
        chunk_len=target_chunk_len(
                        obj=obj,
                        target_mb=target_mb_per_chunk,
                        chunk_along=dim_to_rechunk,
                        split_latlon=chunk_latlon,
                        chunksize_latlon=chunksize_latlon)
        if not chunk_latlon:
            dict_rechunk[var]={d:chunk_len if d==dim_to_rechunk else -1 for d in data.dims}
        else:
            dict_rechunk[var]={d:chunksize_latlon if d in latlondims_ds else \
                            chunk_len if d==dim_to_rechunk else -1 for d in data.dims}
    #---rechunk---
    #rechunk the data variables
    ds_rechunked = {
        v: ds[v].chunk(dict_rechunk[v])
        for v in dict_rechunk.keys()
    }
    ds_rechunked = xr.Dataset(ds_rechunked, coords=ds.coords)

    #rechunk the coordinates
    for coord in ds.coords:
        if dim_to_rechunk in ds[coord].dims:
            ds_rechunked[coord]=ds[coord].chunk({dim_to_rechunk:\
            dict_rechunk[next(iter(dict_rechunk.keys()))][dim_to_rechunk]})
        if chunk_latlon:
            for d in latlondims_ds:
                if d in ds[coord].dims:
                    ds_rechunked[coord]=ds_rechunked[coord].chunk({d:chunksize_latlon})
    #drop current encoding of the chunks
    for v in ds_rechunked.variables:
        ds_rechunked[v].encoding.pop("chunks", None)

    return ds_rechunked

def saveXrToZarr(
            ds:xr.Dataset,
            savedir:str,
            filename:str,
            rechunk:bool=True,
            dim_to_rechunk:str="file_number", 
            target_mb_per_chunk:int=100,
            chunk_total_ds:bool=False,
            chunk_latlon:bool=False,
            chunksize_latlon:int=192
            ) -> None:
    ''' 
    Function to save a xarray dataset to a zarr file.
    Optionally rechunk the dataset along a specific dimension to target a specific chunk size in MB
    before saving.
    '''
    #---error handling---
    if not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.Dataset")
    
    if rechunk and dim_to_rechunk not in ds.dims:
        raise ValueError(f"Dimension '{dim_to_rechunk}' not found in dataset dimensions {ds.dims}")
    
    if not isinstance(target_mb_per_chunk, int) or target_mb_per_chunk <= 0:
        raise ValueError("target_mb_per_chunk must be a positive integer")
    
    if not filename.endswith(".zarr"):
        raise ValueError("Filename must end with '.zarr'")
    
    #---make sure everything is single to reduce size---
    for var in ds.data_vars:
        if re.search(r"lat|lon|x|y", var.lower()):
            continue
        if ds[var].dtype != np.float32:
            ds[var] = ds[var].astype(np.float32)
        elif ds[var].dtype == np.int64:
            ds[var] = ds[var].astype(np.float32)

    #---rechunk the dataset---
    if rechunk:
        ds=rechunk_zarr(
                    store=ds,
                    dim_to_rechunk=dim_to_rechunk, 
                    target_mb_per_chunk=target_mb_per_chunk,
                    chunk_total_ds=chunk_total_ds,
                    chunk_latlon=chunk_latlon,
                    chunksize_latlon=chunksize_latlon
                    )
        
    #---save to zarr---
    os.makedirs(savedir, exist_ok=True)
    print(f"Saving the file to: {os.path.join(savedir,filename)}")
    ds.to_zarr(os.path.join(savedir,filename), 
                  mode="w", 
                  consolidated=True,
                  zarr_format=2
                  )
    print(f"Saved the file to: {os.path.join(savedir,filename)}")