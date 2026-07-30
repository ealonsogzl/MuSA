''' 
Script to create an xr dataset with the results of SNOWCLIM results.
Note that the results are for the measurement sites in the 100m grid, which stil contains duplicated data!

author: Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
'''

# --- modules ---
import os, sys, argparse, re, subprocess
from pathlib import Path
import xarray as xr
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOTDIR = Path(
    subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=SCRIPT_DIR,
        text=True,
    ).strip()
)
sys.path.insert(0, str(ROOTDIR))

from utils.OperationsXrDatasets import saveXrtoNetCDF
from utils.SelectDataFromFolders import SelectDataTiles

# --- functions ---
def parse_args():
    parser = argparse.ArgumentParser(description='Create a dataset with the results of SNOWCLIM.')
    parser.add_argument('--tileList', 
                        type=str, 
                        default='/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/auxdata/mountain_tiles/Alps_tiles.txt',
                        help='Path to the text file containing the list of tiles to process.')
    parser.add_argument('--datasetSDmeasurements',
                        type=str,
                        default='/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/measurements/insitu/Alps_dataset_SD.nc',
                        help='Path to the dataset containing the SD measurements.')
    parser.add_argument('--date_ini',
                    type=str,
                    default='2015-09-01',
                    help='Initial date for the dataset.')
    parser.add_argument('--date_end',
                    type=str,
                    default='2024-08-31',
                    help='Final date for the dataset.')
    parser.add_argument('--savedir',
                        type=str,
                        default='/kyukon/data/gent/vo/000/gvo00090/vsc44965/Doctoraat/Python/Machine_learning/Traditional_MLA/sd/MuSA/DATA',
                        help='Directory where the output dataset will be saved.')
    parser.add_argument('--datadirSnowclimResults',
                        type=str,
                        default='/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/Snowclim/SC_output/noDA/Alps/SC_sim',
                        help='Directory where the SNOWCLIM results are stored.')

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    # Load the dataset with the SD measurements
    ds_measurements = xr.open_dataset(args.datasetSDmeasurements)

    # Select the tiles to process
    tiles=pd.read_csv(args.tileList)

    # Create the dataset with the SNOWCLIM results
    ncores=len(os.sched_getaffinity(0))
    print(f"Number of cores available: {ncores}")

    dsValidationSnowclim=SelectDataTiles(tiles=tiles,
                dates=pd.date_range(args.date_ini,args.date_end,freq="D"),
                dsMeas=ds_measurements,
                datadirVariable=args.datadirSnowclimResults,
                num_cores=ncores
                )

    # Save the dataset to NetCDF
    region=re.search(r"([a-zA-Z]{1,}(.+?))(_tiles)", 
                     os.path.basename(args.tileList)).group(1)

    saveXrtoNetCDF(ds=dsValidationSnowclim, 
                savedir=args.savedir,
                filename=f"Dataset_validation_Snowclim_{region}.nc")

if __name__ == "__main__":
    main()