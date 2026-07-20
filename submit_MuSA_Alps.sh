#!/bin/bash
: '
Shell script to submit the MuSA runs for the Alps on a specific cluster.
The shell script performs parallel job submission over processing tiles.
Each tile is processed in a separate job, which calls the runMuSAtile.sh script!

author: Lucas Boeykens - lucas.boeykens@ugent.be lucas.boeykens@kuleuven.be
'

#--- inputs ---
cluster=${1:-"skiddo"}
modelOnlySites=${2:-True}
remove_output_cells=${3:-True}
date_ini=${4:-"2015-09-01 00:00"}
date_end=${5:-"2024-08-31 23:00"}
rootdirMuSAruns=${6:-"/kyukon/data/gent/vo/000/gvo00090/vsc44965/Doctoraat/Python/Machine_learning/Traditional_MLA/sd/MuSA/DATA/"}

# -- hard coded ---
tiledir="/kyukon/data/gent/vo/000/gvo00090/vsc44965/\
Doctoraat/SNOWSHOP/Tile_lists/"

region="Alps"

ROOTDIR=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
submitJobsScript="${ROOTDIR}/submission/runMuSAtile.sh"

cpus=24
if [ "$modelOnlySites" == "True" ]; then
    time="01:00:00"
else
    time="12:00:00"
fi


#--- submit the jobs ---
# get the number of tiles to process
file=$(find $tiledir -name "${region}_*.txt" | head -n 1)
ntiles="$(wc -l <"$file")"

# submit
job=$(sbatch --job-name=MuSArun_Alps \
  --output=logs/output_MuSArun_Alps_%A.log \
  --error=logs/error_MuSArun_Alps_%A.log \
  --nodes=1 \
  --cpus-per-task=$cpus \
  --time=$time \
  --cluster=${cluster} \
  --array=0-$((ntiles-1)) \
  --wrap="$jobscript \${SLURM_ARRAY_TASK_ID} \
                        $rootdirMuSAruns \
                        $date_ini \
                        $date_end \
                        $modelOnlySites \
                        $remove_output_cells"
                        )

