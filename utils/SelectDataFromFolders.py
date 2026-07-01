#---modules---
import os,glob ,re, tarfile
import pandas as pd

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
    
