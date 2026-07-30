#---modules---
import os,glob ,re, tarfile, gc
import pandas as pd
import xarray as xr 
from joblib import Parallel, delayed

#---custom functions---
def extract_files_to_read(
        dir:str=None,
        tiles:pd.DataFrame=None,
        dates:list[pd.Timestamp]=None,
        verbose:bool=True
        )-> list[str]|list[tuple[str,str]]:
    ''' 
    Function to read in files from a parent dir based on a set of given dates and given tiles
    Inputs:
    - dir: parent directory where the data files are stored
    - dates: list of dates to read files for
    - tiles: dataframe with tx and ty columns indicating the tiles to read

    The function assumes that all subdirecties correspond with tiles and that all subdirectories have the same extentions (.zarr, .tar, .tar.gz, or None)

    Output:
    - list[str]: if the files are in a regular directory and/or regular netCDF-files
    - list[tuple[str,str]]: if the data are tarred or if the data are stored in zarr format
    '''
    #TODO: implement tarred zarr folders

    #---hard coded---
    date_pattrn=r"\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])"
    dir=dir if dir.endswith("/") else dir+"/"

    #---list up the directories or files---
    dirs=glob.glob(dir+"*")
    if not dirs:
        print("no subdirectories nor files found!") if verbose else None
        return None

    #---extract those for the folders---
    dirs=[d for d in dirs if any(re.search(rf"y{ty:03d}x{tx:03d}", d) 
                                for ty,tx in zip(tiles["ty"], tiles["tx"]))]
    if not dirs:
        print("no subdirectories or files found for the given tiles!") if verbose else None
        return None

    #---extract files to read---
    #regular files in a directory
    if os.path.isdir(dirs[0]) and not os.path.basename(dirs[0]).endswith(".zarr"):
        print("The data are stored as regular files in directories") if verbose else None
        #extract the files in the directories
        files_to_read=[f for sub in [glob.glob(d+"/*nc") if os.path.isdir(d) 
                                    else [d] for d in dirs] 
                        for f in sub]
        #filter on dates
        files_to_read=[f for f in files_to_read if dates is None or \
                       any(date.strftime('%Y%m%d') in f for date in dates)]
        if files_to_read:
            print("Date information found in the file names.") if verbose else None
        else:
            print("No date information found in the file names.") if verbose else None
            files_to_read=[f for sub in [glob.glob(d+"/*nc") if os.path.isdir(d) 
                                        else [d] for d in dirs] 
                            for f in sub]
            files_to_read=[f for f in files_to_read \
                if any(re.search(rf"y{ty:03d}x{tx:03d}", f) \
                    for ty,tx in zip(tiles["ty"], tiles["tx"]))]
    
    #specific files/extentions: NO REGULAR DIRECTORIES
    else:
        #get the extention -> determines further steps
        ext=re.search(r"\.(.+?)$", os.path.basename(dirs[0])).group(0)
        #tarred files
        if re.search(r".tar|.tar.gz", ext):
            print("The data are tarred") if verbose else None
            unzip="r" if ext==".tar" else "r:gz"

            files_to_read=[]
            for i,dir in enumerate(dirs):
                with tarfile.open(dir, unzip) as tr:
                    #list up the files in the tar file
                    tarred_files=tr.getmembers()
                    #check if there is any date info in the tarred files
                    if any(re.search(date_pattrn, t.name) for t in tarred_files):
                        if i==0:
                            print("Date information found in the file names.") if verbose else None
                        #get the files for the date
                        files_to_read.extend([(dir,f) for f in tarred_files if dates is None or \
                                            any(date.strftime('%Y%m%d') in f.name for date in dates)])
                    else:
                        if i==0:
                            print("No date information found in the file names.") if verbose else None
                        #get the files for the tiles
                        files_to_read.extend([(dir,f) for f in tarred_files \
                            if any(re.search(rf"y{ty:03d}x{tx:03d}", f.name) \
                                for ty,tx in zip(tiles["ty"], tiles["tx"]))]) 
        #zarr stores
        elif ext==".zarr":
            print("The data are stored in zarr format") if verbose else None
            files_to_read=[(d, list(dates)if dates is not None else None) for d in dirs]
        #regular files
        elif ext in [".nc", ".nc4", ".h5", ".hdf5"]:
            print("The data are stored as regular files") if verbose else None
            #check for date info in the file names
            if any(re.search(date_pattrn, f) for f in dirs):
                print("Date information found in the file names.") if verbose else None
                files_to_read=[f for f in dirs if dates is None or \
                               any(date.strftime('%Y%m%d') in f for date in dates)]
            else:
                print("No date information found in the file names.") if verbose else None
                files_to_read=[f for f in dirs \
                    if any(re.search(rf"y{ty:03d}x{tx:03d}", f) \
                        for ty,tx in zip(tiles["ty"], tiles["tx"]))]
                
    return files_to_read
    
def SelectDataDate(date_info, verbose:bool=False) -> pd.DataFrame:
    ''' 
    Function that selects the data of a dataset with point measurements for a given date and tiles.

    arguments:
    - date_info: tuple with (tiles, date, dsMeas, datadirVariable)
        - tiles: dataframe with the tiles to select
        - date: date to select
        - dsMeas: dataset with the point measurements. must contain the dimensions "site" and "date" and a variable with snow depth (e.g. "sd_clean" or "sd")
        - datadirVariable: directory where the data is stored
    
    '''
    #---inputs---
    tiles, date, dsMeas, datadirVariable, index = date_info

    #---get the dimensions and variable names of dsMeas---
    #get the site and date id dimension
    siteID=next(d for d in dsMeas.dims if 
                re.search(r"site|id", d.lower()))
    timeID=next(d for d in dsMeas.dims if 
                re.search(r"time|date", d.lower()))
    #get the snow depth variable
    try:
        snowVar=next(v for v in dsMeas.data_vars if 
                    re.search(r"sd_clean", v.lower()))
    except StopIteration:
        try:
            snowVar=next(v for v in dsMeas.data_vars if 
                        re.search(r"sd", v.lower()))
        except StopIteration:
            raise ValueError("No variable found in the dataset that matches "
                            "snow depth (e.g. 'sd_clean' or 'sd')")

    #---check the type of folders---
    dtype=next(iter(glob.glob(os.path.join(datadirVariable,"*"))))
    if re.search(r".tar(.gz)?$", dtype):
        unzip="r" if re.search(r".tar$", dtype) else "r:gz"
        dtype="tarred"
    else:
        dtype="unzipped"

    #---make dataframe with the features for all tiles---
    dfFeaturesTiles=[]
    for tx,ty in zip(tiles.tx,tiles.ty):
        print(f"Processing tile y{ty:03d}x{tx:03d}") if verbose else None

        #---check the tiledir---
        try:
            tiledir=next(iter(glob.glob(os.path.join(datadirVariable,
                                                f"*y{ty:03d}x{tx:03d}*"))))
        except StopIteration:
            print(f"No folder found for tile y{ty:03d}x{tx:03d}") if verbose else None
            continue

        # check for subdirectory containing the nc-data
        subdirs=glob.glob(os.path.join(tiledir, "*"))
        if any(os.path.isdir(d) for d in subdirs):
            # find the subdirectory that contains the nc-files
            subdir=[d for d in subdirs if any(glob.glob(os.path.join(d, "*.nc")))]
            if len(subdir)>1:
                raise NotImplementedError("Code only implemented for checking a single subdirectory with nc-files in the tiledir!" \
                "Please revise the directory structure of the tile directory!")
            else:
                tiledir=subdir[0]

        #---check if auxiliary data---
        check=next(iter(glob.glob(os.path.join(tiledir, "*.nc")))) if dtype=="unzipped" \
            else next(iter(tarfile.open(tiledir, unzip).getnames())) 
        if not re.search(r"(.+)?\d{8}(.+)?.nc", os.path.basename(check)):
            #Get sites in the tile
            indexTile=(dsMeas["tx"]==tx) & (dsMeas["ty"]==ty)
            dsMeasTile=dsMeas.isel({siteID: indexTile})
            #open the files
            df_aux=[]
            if dtype=="tarred":
                with tarfile.open(tiledir, unzip) as tar:
                    files=[f for f in tar.getnames() if re.search(r".nc$", f)]
                    for file in files:
                        file=tar.extractfile(file)
                        dsFeaturesTile=xr.open_dataset(file).\
                            sel(lat=dsMeasTile["lat"], 
                                lon=dsMeasTile["lon"], 
                                method="nearest").load()
                        dfFeaturesTile=dsFeaturesTile.to_dataframe().\
                                    dropna(subset=list(dsFeaturesTile.data_vars.keys()), 
                                            how="all").\
                                    drop(columns=["lat", "lon"], axis=1)   
                        df_aux.append(dfFeaturesTile)
                        
            elif dtype=="unzipped":
                files=glob.glob(os.path.join(tiledir, "*.nc"))
                for file in files:
                    dsFeaturesTile=xr.open_dataset(file).\
                        sel(lat=dsMeasTile["lat"], 
                        lon=dsMeasTile["lon"], 
                        method="nearest").load()
                    dfFeaturesTile=dsFeaturesTile.to_dataframe().\
                                dropna(subset=list(dsFeaturesTile.data_vars.keys()), 
                                        how="all").\
                                drop(columns=["lat", "lon"], axis=1)   
                    df_aux.append(dfFeaturesTile)
            df_aux=pd.concat(df_aux,axis=1)
            dfFeaturesTiles.append(df_aux)
            continue

        #---extract the sites and dates from the tile---
        #Get sites in the tile
        indexTile=(dsMeas["tx"]==tx) & (dsMeas["ty"]==ty)
        dsMeasTile=dsMeas.isel({siteID: indexTile})
        #extract the dates for which measurements are available
        indexDate=~dsMeasTile[snowVar].isnull().all(dim=siteID)
        dsMeasTile=dsMeasTile.isel({timeID: indexDate})
        #check if the date is in the tile
        if date not in dsMeasTile[timeID].values:
            print(f"No measurements found for date {date.strftime('%Y-%m-%d')} in tile y{ty:03d}x{tx:03d}") if verbose else None
            continue

        #---open the file and select the data for the dates and sites in the tile---
        if dtype=="tarred":
            with tarfile.open(tiledir, unzip) as tar:
                file=next((f for f in tar.getnames() if 
                        re.search(date.strftime("%Y%m%d"), f)), None)
                if file is None:
                    print(f"No file found for date {date.strftime('%Y%m%d')}") if verbose else None
                    continue
                file=tar.extractfile(file)
                ds=xr.open_dataset(file)
                dsFeaturesTile=ds.sel(lat=dsMeasTile["lat"], 
                                    lon=dsMeasTile["lon"], 
                                    method="nearest").load()
        elif dtype=="unzipped":
            file=next(iter(glob.glob(os.path.join(tiledir, 
                                                f"*{date.strftime('%Y%m%d')}*"))), 
                        None)
            if file is None:
                print(f"No file found for date {date.strftime('%Y%m%d')}") if verbose else None
                continue
            dsFeaturesTile=xr.open_dataset(file).\
                sel(lat=dsMeasTile["lat"], 
                lon=dsMeasTile["lon"], 
                method="nearest").load()
        #check for the date
        if not timeID in dsFeaturesTile.coords:
            dsFeaturesTile=dsFeaturesTile.expand_dims(timeID).\
                assign_coords({timeID:[date]})
        #make a dataframe
        dfFeaturesTile=dsFeaturesTile.to_dataframe().reset_index().\
            dropna(subset=list(dsFeaturesTile.data_vars.keys()), 
                    how="all")
        #check if satellite data
        SatMatch=re.search(r"(\d{3})_(A|D)", file.name if dtype=="tarred" else file)
        if SatMatch:
            orbit=SatMatch.group(1)
            track=SatMatch.group(2)
            #add to dataframe
            dfFeaturesTile["orbit"]=orbit
            dfFeaturesTile["track"]=track
        #add to overall dataframe
        dfFeaturesTiles.append(dfFeaturesTile)

        del dsFeaturesTile, dfFeaturesTile
    
    gc.collect() if index%100==0 else None
    if dfFeaturesTiles:
        return pd.concat(dfFeaturesTiles, axis=0)
    else:
        return None
    
def SelectDataTiles(
        tiles:pd.DataFrame=None,
        dates:list[pd.Timestamp]=None,
        dsMeas:xr.Dataset=None, 
        datadirVariable:str=None,
        num_cores:int=len(os.sched_getaffinity(0))
        ) -> pd.DataFrame:
    '''
    Function to select data at measurement sites if data is stored as netCDF-files!
    The function recognizes if auxdata is provided. If so, not all dates are 
    '''
    #---get the dimensions and variable names of dsMeas---
    #get the site and date id dimension
    siteID=next(d for d in dsMeas.dims if 
                re.search(r"site|id", d.lower()))
    timeID=next(d for d in dsMeas.dims if 
                re.search(r"time|date", d.lower()))

    #---make dataframe with the features for all tiles---
    #make the dates
    date_info=[(tiles, date, dsMeas, datadirVariable, i) \
        for i, date in enumerate(dates)]
    #check for auxdata -> if available, then no need to pool, else pool over the dates
    dfFeaturesTiles=SelectDataDate(date_info[0])
    if dfFeaturesTiles is None or any(re.search(r"date|time", d.lower()) for d in dfFeaturesTiles.columns):
        dfFeaturesTiles=Parallel(n_jobs=num_cores, verbose=10)(
            delayed(SelectDataDate)(info) 
            for info in date_info
            )
        dfFeaturesTiles=pd.concat(dfFeaturesTiles, axis=0)
        
    #---make netCDF ready dataset---
    if any(c in ["track", "orbit"] for c in dfFeaturesTiles.columns):
        dfFeaturesTiles=dfFeaturesTiles.set_index(keys=[siteID, timeID, "track"])
    else:
        dfFeaturesTiles=dfFeaturesTiles.set_index(keys=[siteID, timeID])
    if "ind" in dfFeaturesTiles.columns:
        dfFeaturesTiles=dfFeaturesTiles.drop(columns=["ind"],axis=1)

    return dfFeaturesTiles.to_xarray()