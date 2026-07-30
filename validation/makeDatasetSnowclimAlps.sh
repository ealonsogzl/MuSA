#!/bin/bash
cluster=${1:-doduo} #str, e.g. doduo

# --- hard coded ---
tileList="/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/auxdata/mountain_tiles/Alps_tiles.txt"
savedir="/kyukon/data/gent/vo/000/gvo00090/vsc44965/Doctoraat/Python/Machine_learning/Traditional_MLA/sd/MuSA/DATA"
datadirSnowclimResults='/kyukon/data/gent/vo/000/gvo00090/SNOWSHOP/Snowclim/SC_output/noDA/Alps/SC_sim'

pythonpath="/kyukon/data/gent/vo/000/gvo00090/\
vsc44965/Conda/envs/MuSAenv/bin/python"

# --- submit job ---
scriptdir="$(dirname "$0")"

job=$(sbatch --job-name=MakeDatasetSnowclim \
    --output=logs/output_MakeDatasetSnowclim_%A.log \
    --error=logs/errors_MakeDatasetSnowclim_%A.log \
    --nodes=1 --cpus-per-task=12 --time=06:00:00 \
    --cluster=$cluster \
    --wrap="$pythonpath $scriptdir/MakeDatasetSnowclimResults.py \
                                        --tileList $tileList \
                                        --savedir $savedir \
                                        --datadirSnowclimResults $datadirSnowclimResults"
                                        )
