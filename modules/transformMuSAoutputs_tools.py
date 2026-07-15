#---modules---
import os,re,glob,sys, shutil
import pandas as pd 
import xarray as xr 
sys.path.append(os.getcwd())
import modules.internal_fns as ifn
import dask.array as da
import numpy as np
import modules.internal_fns as ifn
from joblib import Parallel, delayed
from utils.OperationsXrDatasets import transpose_dataset, saveXrToZarr, saveXrtoNetCDF
from pathlib import Path

#---functions---
def _ReturnSpecificsForcingArray(args:dict) -> tuple[tuple, dict, dict]:
    #---get the forcing files---
    forcings=ifn.check_forcings_timerange(
        date_ini=args.date_ini,
        date_end=args.date_end,
        forcing_dir=args.nc_forcing_path,
        verbose=False
        )

    #---create an empty template---
    if isinstance(forcings, str):
        ds_forcings=xr.open_zarr(forcings, consolidated=True)
        
        shape_array=tuple(ds_forcings.sizes.values())
        dims_array=dict(ds_forcings.sizes)

        coords_array=dict(ds_forcings.coords)

    elif isinstance(forcings, list):
        #get the amount of days
        ndays=len(forcings)

        #get the amount of time steps in the first forcing file
        nc_forcing=transpose_dataset(xr.open_dataset(forcings[0], chunks={}))
        time_dim=next((d for d in nc_forcing.dims if re.fullmatch(r'time|t', d)), None)
        if time_dim is None:
            raise ValueError("No time dimension found in the dataset.")
        ntimes=nc_forcing.sizes[time_dim]
        
        shape_array=(ndays, ntimes) + tuple(nc_forcing.sizes[d] for d in nc_forcing.dims if d != time_dim)

        #get the dimensions of the forcing files
        dims_array={"date":ndays}
        dims_array.update({d:nc_forcing.sizes[d] for d in nc_forcing.dims})

        #get the coords
        coords_array={"date":xr.DataArray(pd.to_datetime([re.search(r"\d{8}", os.path.basename(f)).group() for f in forcings], 
                                                    format='%Y%m%d'), 
                                    dims=["date"], 
                                    name="date")}
        coords_array.update({d:nc_forcing.coords[d] for d in nc_forcing.dims})
    
    return shape_array, dims_array, coords_array


def create_template_zarr(
        args:dict,
        vars_to_save:list[str]=["snd", "SWE", "fSCA"],
        ) -> str:
    ''' 
    Function to create a template zarr file. Note that the mean per day is taken for the vars to save.
    '''
    # get the shape of the array and the dimensions and coordinates for the forcing data
    shape_array, dims_forcings, coords_forcings=_ReturnSpecificsForcingArray(args)
    
    # remove the time dimension from the shape and dims and coords
    shape_array = tuple(v for k,v in dims_forcings.items() if not re.fullmatch(r'time|t', k))
    dims_forcings = {k: v for k, v in dims_forcings.items() if not re.fullmatch(r'time|t', k)}
    coords_forcings = {k: v for k, v in coords_forcings.items() if not re.fullmatch(r'time|t', k)}
    
    # create an empty dataset with the specified variables and dimensions
    ds_template=[]
    for var in vars_to_save:
        arr=da.empty(
            shape=shape_array,
            chunks=tuple(1 if dim in ["lat", "lon"] else -1 \
                         for dim in dims_forcings.keys()),
            dtype=np.float32
        )
        ds_template.append(
            xr.DataArray(
                arr,
                dims=dims_forcings,
                coords=coords_forcings,
                name=var,
            )
        )
    ds_template=xr.merge(ds_template)

    #--save the template to zarr---
    out_path=os.path.join(args.output_path, "resultsSpatialGrid_tmp.zarr")
    ds_template.to_zarr(out_path, mode="w", compute=False)

    return out_path


def WriteCellsToZarr(
        file: str ,
        store_to_write: str,
        args: dict,
        vars_to_save:list[str]=["snd", "SWE", "fSCA"],
    ) -> None:

    #extract lat and lon indices from the filename
    idx_match=re.search(r"(\d{1,3})_(\d{1,3}).pkl", file)
    idx_lat=int(idx_match.group(1))
    idx_lon=int(idx_match.group(2))

    #extract the datetime index from the forcings zarr file
    forcings=ifn.check_forcings_timerange(
        date_ini=args.date_ini,
        date_end=args.date_end,
        forcing_dir=args.nc_forcing_path,
        verbose=False
        )
    with xr.open_zarr(forcings, consolidated=True) as ds_forcings:
        index_datetime=ds_forcings[["date", "time"]].\
            stack(datetime=("date", "time")).to_dataframe().index
        
        lat_idx = ds_forcings["lat"].isel(lat=idx_lat).item()
        lon_idx = ds_forcings["lon"].isel(lon=idx_lon).item()

    # open the file -> using io_read and set index
    cell=ifn.io_read(file)
    cell=cell.set_index(index_datetime)

    # take the mean of the outputs per day
    cell=cell.groupby(cell.index.get_level_values("date"))[vars_to_save].mean()


    # generate an xr dataset from the vars to save
    cell_ds=cell[vars_to_save].to_xarray()

    #add lat, lon info
    cell_ds = cell_ds.expand_dims({"lat": [lat_idx], 
                                "lon": [lon_idx]})

    #save to the zarr
    cell_ds.drop_vars(["date"]).to_zarr(
        store_to_write,
        region={"lat": slice(idx_lat, idx_lat + 1), "lon": slice(idx_lon, idx_lon + 1)},
        mode="r+",
    )

def saveFinalOutputToZarr(
        args:dict,
        vars_to_save:list[str]=["snd", "SWE", "fSCA"],
        removeCells:bool=True
    ) -> None:
    ''' 
    Function to save the finall output to a zarr store.
    
    '''
    #creat the template zarr file
    out_path=create_template_zarr(args, 
                                  vars_to_save=vars_to_save
                                  )

    #write the cells to the zarr file
    cells=glob.glob(os.path.join(args.output_path, "*.pkl*"))
    _ = Parallel(
        n_jobs=-1,
        verbose=10,
        )(
        delayed(WriteCellsToZarr)(
            file=cell,
            store_to_write=out_path,
            args=args,
            vars_to_save=vars_to_save
        )
        for cell in cells
        )

    #rechunk the zarr file to optimize for reading
    save_path=out_path.replace("_tmp.zarr", ".zarr")
    saveXrToZarr(
            ds=xr.open_zarr(out_path, consolidated=True),
            savedir=os.path.dirname(save_path),
            filename=os.path.basename(save_path),
            rechunk=True,
            dim_to_rechunk="date", 
            target_mb_per_chunk=150,
            chunk_total_ds=False,
            chunk_latlon=True,
            chunksize_latlon=192
            )
    
    # remove the temporary zarr file and the individual cell files
    shutil.rmtree(out_path)
    
    if removeCells:
        for cell in cells:
            Path(cell).unlink(missing_ok=True)


def _process_cells_onlysites(
        args:dict,
        dsMeas: xr.Dataset = None,
        vars_to_save:list[str]=["snd", "SWE", "fSCA"]
    ) -> xr.Dataset:
    '''
    Process all cell files in parallel and return one xarray dataset.

    The returned dataset uses `date` and `site` as coordinates/dimensions.
    '''
    if dsMeas is None:
        raise ValueError("`dsMeas` must be provided.")

    #get the list of cell files to process
    cells=glob.glob(os.path.join(args.output_path, "*.pkl*"))

    #get the index
    _, _, coords_forcings=_ReturnSpecificsForcingArray(args)
    index_datetime = pd.MultiIndex.from_product(
        [coords_forcings["date"].values, coords_forcings["time"].values],
        names=["date", "time"],
    )

    #open the mask dataset
    mask_ds = xr.open_dataset(args.nc_maks_path)

    # extract the measurement sites from the output path 
    tile_match = re.search(r"y(\d{3})x(\d{3})", args.output_path)
    ty = int(tile_match.group(1))
    tx = int(tile_match.group(2))

    index=(dsMeas["tx"]==tx) & (dsMeas["ty"]==ty)
    dsMeasTile=dsMeas.where(index, drop=True)
    sites=dsMeasTile["site"].values

    # generate a DataFrame for each site and concatenate them
    df_sites=[]
    for site in sites:
        # print(f"Processing site {site}...")
        # select the site and the coordinates of the site
        dsSite=dsMeas.sel(site=site)


        idx_lat_site=np.argmin(abs(mask_ds["lat"].values-dsSite["lat"].values))
        idx_lon_site=np.argmin(abs(mask_ds["lon"].values-dsSite["lon"].values))

        file=next((c for c in cells if re.search(rf"cell_{idx_lat_site}_{idx_lon_site}.+?", c)), None)
        if file is None:
            print(f"No cell file found for site {site} at indices ({idx_lat_site}, {idx_lon_site}).")
            continue

        # read the cell
        cell = ifn.io_read(file).copy()
        # error handling: check if the length of the cell matches the length of the index
        if len(cell) != len(index_datetime):
            raise ValueError(
                f"File {file} has {len(cell)} rows, but the forcing index has {len(index_datetime)} entries."
            )
        # set the index and take daily means of the cell data
        cell.index = index_datetime
        cell = (
            cell.reset_index()
            .groupby("date", as_index=False)[vars_to_save]
            .mean()
        )
        cell["date"] = pd.to_datetime(cell["date"])

        # get the site information from the measurements dataset
        site_info = (
            dsMeas.isel(site=site)
            .sel(date=slice(cell["date"].min(), cell["date"].max()))
            .to_dataframe()
            .reset_index()
        )
        site_info["site"] = site

        # merge the cell data with the site information
        df_site = pd.merge(cell, site_info, on="date", how="left", suffixes=("", "_meas"))
        df_site["site"] = site
        df_site["idx_lat"] = idx_lat_site
        df_site["idx_lon"] = idx_lon_site

        df_sites.append(df_site)
    df_sites=pd.concat(df_sites).set_index(["date", "site"]).sort_index()
    df_sites=df_sites.rename(columns={v:f"{v}_FSM" for v in vars_to_save})

    return df_sites.to_xarray()


def saveFinalOutputSitesOnly(
        args:dict,
        dsMeas:str="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/measurements/insitu/Alps_dataset_SD.nc",
        vars_to_save:list[str]=["snd", "SWE", "fSCA"],
        removeCells:bool=True
    ) -> None:
    '''
    Function to save the final output cells to a netcdf file, using only the sites from the measurements dataset.
    '''
    #process the cells
    ds_site = _process_cells_onlysites(args, dsMeas=xr.open_dataset(dsMeas), vars_to_save=vars_to_save)

    #save the final dataset to netcdf
    saveXrtoNetCDF(
        ds=ds_site,
        savedir=args.output_path,
        filename="resultsSitesOnly.nc",
        )
    
    # remove the individual cell files
    if removeCells:
        cells=glob.glob(os.path.join(args.output_path, "*.pkl*"))
        for cell in cells:
            Path(cell).unlink(missing_ok=True)