#script containing code to read in individual focing files (*.nc-files), and
#convert them to a single zarr file, which is then used as input for the MuSA model.

# author: Lucas.boeykens@ugent.be, lucas.boeykens@kuleuven.be

#----modules---
import numpy as np
import xarray as xr
import pandas as pd
import os, sys, re, tarfile, shutil
sys.path.append((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.SelectDataFromFolders import extract_files_to_read 
from utils.OperationsXrDatasets import transpose_dataset, saveXrToZarr
from pathlib import Path
import dask.array as da
from joblib import Parallel, delayed

#---custom functions---
def _makeEmptyTemplateZarrForcings(
    rootdirForcings:str="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/Forcings/tiles/3hourly",
    tx:int=201, 
    ty:int=35,
    date_ini:str=None,
    date_end:str=None,
    savedir:str=os.getcwd(),
    filename:str="template_forcings.zarr"
    ) -> tuple[str, list[str], list[pd.DatetimeIndex]]:
    ''' 
    Function to create an empty template zarr for the forcings
    of a specific tile (tx, ty) and save it to a specified directory.
    The template will have the same dimensions and coordinates as the first file found for the specified tile
    and will be filled with NaN values. The function returns the path to the saved zarr file, the list of files read, 
    and the corresponding dates.

    CURRENTLY ONLY ONE TILE AT THE TIME IS SUPPORTED
    '''

    #---get the amount of files to read in---
    if date_ini is None or date_end is None:
        dates=None
    else:
        dates=pd.date_range(start=pd.Timestamp(date_ini), 
                            end=pd.Timestamp(date_end), 
                            freq='D')

    files_to_read=sorted(extract_files_to_read(
            dir=rootdirForcings,
            tiles=pd.DataFrame({"tx":[tx], "ty":[ty]}),
            dates=dates,
            verbose=False #HARD CODED: set to True if you want to know the type of files to read in
            ))
    #extract the amount of files to read in
    nfiles=len(files_to_read)

    #---make a template---
    ds0=transpose_dataset(xr.open_dataset(files_to_read[0], chunks={}))
    #extrac the forcings
    forcings = [
        v for v in ds0.data_vars
        if not re.search(r"offset", v, re.IGNORECASE) #HARD CODED with offset
    ]
    # Extract date coordinate from filenames
    dates=[
        pd.to_datetime(re.search(r"\d{4}\d{2}\d{2}", Path(f).name).group())
        for f in files_to_read
    ]
    #sort both files and dates together based on the dates
    files_to_read, dates = zip(*sorted(zip(files_to_read, dates), key=lambda x: x[1]))
    dates = pd.DatetimeIndex(dates)

    # Keep all original dims, including time
    base_dims = dict(ds0[forcings].sizes)
    # Add new date dimension in front
    dims_template = {
        "date": nfiles,
        **base_dims,
    }
    chunks_template = {
        "date": 1,
        **{d: -1 for d in base_dims}
    }
    coords = {
        "date": dates,
    }
    # Copy all coords that belong to original dimensions, including time, lat, lon, x, y
    for c in ds0.coords:
        coord_dims = ds0[c].dims
        if all(d in base_dims for d in coord_dims):
            coords[c] = ds0[c]
    
    template_vars = {}
    for var in forcings:
        arr = da.empty(
            shape=tuple(dims_template.values()),
            dtype=np.float32,
            chunks=tuple(chunks_template[d] for d in dims_template),
        )

        template_vars[var] = xr.DataArray(
            arr,
            dims=tuple(dims_template.keys()),
            coords=coords,
            name=var,
            attrs=ds0[var].attrs,
        )
    ds_template = xr.Dataset(template_vars, attrs=ds0.attrs)

    #---save the template to zarr---
    os.makedirs(savedir, exist_ok=True)
    if date_ini is None or date_end is None:
        filename=filename.replace(".zarr", "_tmp.zarr")
    else:
        filename=filename.replace(".zarr", f"_{pd.Timestamp(date_ini).strftime('%Y%m%d')}_{pd.Timestamp(date_end).strftime('%Y%m%d')}_tmp.zarr")

    ds_template.to_zarr(os.path.join(savedir, filename),
                        mode="w", 
                        compute=False)

    return os.path.join(savedir, filename), files_to_read, dates

def ConvertForcingIntegerToFloatFile(
        file_to_read:str|tuple[str,str]=None,
        verbose:bool=True
    ) -> xr.Dataset:
    ''' 
    Function to reads in a netCDF/tarred/zarr file, 
    converts the forcings from integer to float based on the offset values, 
    and returns an xarray dataset.

    Note: The function assumes that the time dimension for the files/stores is date!
    '''

    #---helper function---
    def _convertForcings(
            ds:xr.Dataset=None
        ) -> xr.Dataset:
        ''' 
        Helper function to convert the forcings from integer to float based on the offset values.
        '''
        #get the forcing names #HARD CODED: offset is in the file name of the forcings
        forcings=[d for d in ds.data_vars if not re.search(r"offset", d, re.IGNORECASE)]
        offsets=[d for d in ds.data_vars if re.search(r"offset", d, re.IGNORECASE)]

        #access the metadata
        metadata={k:v.attrs for k,v in ds.data_vars.items() if k in forcings}

        #convert
        ds_converted=ds[forcings].copy()
        for forcing, offset in zip(forcings, offsets):
            #convert the forcing to float based on the offset values
            da_converted=ds[forcing]/ds[offset].isel(aux=1)+ds[offset].isel(aux=0)
            da_converted=da_converted.drop_vars("aux")
            #add to the cobnverted forcings to the dataset
            ds_converted[forcing]=da_converted
            
        for k in ds_converted.data_vars.keys():
            if k in metadata:
                ds_converted[k].attrs=metadata[k]

        return ds_converted

    #---hard coded---
    date_pattrn=r"\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])"

    #---check if files are provided---
    if not file_to_read:
        print("No file to read.") if verbose else None
        return None

    #---check the dtype of the file -> read in the file---
    if isinstance(file_to_read, tuple):
        #case for a tarred file
        if re.search(r".tar|.tar.gz", file_to_read[0]):
            print("The file to read is a tarred file.") if verbose else None

            #open the tar file
            unzip="r" if re.search(r".tar$", file_to_read[0]) else "r:gz"
            with tarfile.open(file_to_read[0], unzip) as tr:
                #extract the file object
                file_obj=tr.extractfile(file_to_read[1])
                #open the dataset
                ds_feature=xr.open_dataset(file_obj).load()
                ds_feature=transpose_dataset(ds=ds_feature)
        #case for a zarr store
        elif re.search(r".zarr", file_to_read[0]):
            print("The file to read is a zarr store.") if verbose else None
            ds_temp=xr.open_zarr(file_to_read[0])
            if file_to_read[1] is not None:
                date_dim=next(d for d in ds_temp.dims if re.search(r"date|time", d))
                #check if the date is in the store, else return None
                try:
                    ds_temp=ds_temp.sel({date_dim: file_to_read[1]})
                    ds_temp=ds_temp.rename_dims({date_dim:"date"}) if date_dim!="date" \
                        else ds_temp
                    ds_feature=transpose_dataset(ds=ds_temp)
                except:
                    ds_feature=None
            #combine the datasets
            if ds_feature is None:
                print("No file to read.") if verbose else None
                return None
    #if not a tuple, then it is a regular file to read
    else:
        print("The file to read is a regular file.") if verbose else None
        #read in files with open mfdataset -> small files assumed
        ds_temp=xr.open_dataset(file_to_read, chunks={})
        ds_temp=transpose_dataset(ds=ds_temp)
        ds_feature=ds_temp

    #---convert the forcings from integer to float based on the offset values---
    ds_feature=_convertForcings(ds_feature)

    #---ADDED: specific conversions to make the forcings compatible for MuSA---
    #TODO: make this part non-hard coded
    time_dim=next((d for d in ds_feature.dims if re.fullmatch(r"time", d)), None)
    timeDim=ds_feature.sizes[time_dim] if time_dim else 1
    nTimeSteps=24//timeDim
    
    ds_feature["P"]=ds_feature["P"]*1000 #convert from m to mm
    ds_feature["SWd"]=ds_feature["SWd"]*nTimeSteps*1000 #convert from kJ/m2/h to J/m2/nTimesteps h
    ds_feature["LWd"]=ds_feature["LWd"]*nTimeSteps*1000 #convert from kJ/m2/h to J/m2/nTimesteps h


    #---expand the date dimension if it is not present in the dataset---
    if re.search(date_pattrn, file_to_read):
        date_index=np.unique(pd.to_datetime([re.search(date_pattrn, file_to_read).group()]))
        date_dim=next((d for d in ds_feature.dims if re.search(r"date|time", d)), None)
        # if no date dimension is found, create a new date dimension and assign the date index to it
        if not date_dim:
            ds_feature=ds_feature.expand_dims(date=pd.Index(date_index, name='date'))
        #if a date dimension is found, check if it is of datetime type, else keep the original coord (e.g. for hourly files)
        else:
            check_datedim=np.issubdtype(ds_feature[date_dim].values.dtype, np.datetime64)
            #only update the date coord (which can be time or date) if it is of datetime type, else keep the original coord (e.g. for hourly files)
            if date_dim != "date" and not check_datedim:
                #update the date coord to match the files read
                ds_feature=ds_feature.expand_dims(
                    date=pd.Index(date_index, name='date')
                    )
                # ds_feature=ds_feature.expand_dims(date=ds_feature[date_dim])
            elif date_dim != "date" and check_datedim:
                ds_feature=ds_feature.rename_dims({date_dim:"date"})
                ds_feature=ds_feature.assign_coords(
                    date=pd.Index(date_index, name='date')
                    )

    return ds_feature

def TransformForcingsFileWriteToZarr(
        store:str=None,
        file_to_read:str=None,
        index:int=0
    ) -> None:
    ''' 
    Function that based on an existent zarr store, and a forcing file to read:
        - open the file to read
        - convert the forcings from integer to float based on the offset values
        - transpose the dataset to match the zarr store
        - drop the lat, lon and time variables (if they exist)
        - write the transformed dataset to the zarr store at the specified index
    '''
    #read in the file to read and convert the forcings from integer to float based on the offset values
    ds_transformed=ConvertForcingIntegerToFloatFile(file_to_read=file_to_read, verbose=False)

    #transpose the dataset to match the zarr store and drop the lat, lon and time variables (if they exist)
    #why: they are constant and need to be dropped to avoid conflicts when writing to the zarr store
    ds_transformed=ds_transformed.transpose(*list(xr.open_zarr(store, consolidated=True).sizes.keys())).\
                                    drop_vars(
                                                ["lat", "lon", "time"],
                                                errors="ignore")

    #write the transformed dataset to the zarr store at the specified index
    ds_transformed.to_zarr(
        store,
        region={"date": slice(index, index + 1)},
        mode="r+",
    )
    del ds_transformed

def CreateZarrTransformedForcings(
    rootdirForcings:str="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/Forcings/tiles/3hourly",
    tx:int=201, 
    ty:int=35,
    date_ini:str=None,
    date_end:str=None,
    savedir:str=os.getcwd(),
    filename:str="test.zarr",
    ncores:int=len(os.sched_getaffinity(0)),
    ncores_min:int=8     
    ) -> None:
    ''' 
    Function that creates a zarr store with converted forcings from integer to float based on the offset values for a specific tile (tx, ty).
    The function will:
        - create an empty template zarr store with the same dimensions and coordinates as the first file found for the specified tile
        - read in the files for the specified tile, convert the forcings from integer to float based on the offset values, and write the transformed data to the zarr store
        - rechunk the zarr store to have a target chunk size of 150 MB per chunk along the date dimension
    '''
    #---error handling---
    ncores=max(ncores_min, ncores) #set to minimum of 8 cores

    #---make the template zarr store---
    store_tmp, files_to_read,_ =_makeEmptyTemplateZarrForcings(
        rootdirForcings=rootdirForcings,
        tx=tx,
        ty=ty,
        date_ini=date_ini,
        date_end=date_end,
        savedir=savedir,
        filename=filename
    )

    #---add the transformed data to the zarr store---
    _ = Parallel(
        n_jobs=ncores,          # or -1, but ncores is often safer for IO
        verbose=10,
    )(
        delayed(TransformForcingsFileWriteToZarr)(
            store=store_tmp,
            file_to_read=file_to_read,
            index=i,
        )
        for i, file_to_read in enumerate(files_to_read)
    )

    #---rechunk the zarr store---
    filename=store_tmp.replace("_tmp.zarr", ".zarr") #go back to the filename without the _tmp suffix
    saveXrToZarr(ds=xr.open_zarr(store_tmp),
             savedir=savedir,
             filename=filename,
             dim_to_rechunk="date",
             target_mb_per_chunk=150
    )

    #--remove the temporary zarr store---
    shutil.rmtree(store_tmp)
