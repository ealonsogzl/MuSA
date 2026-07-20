#!/bin/bash
cluster=${1:-"skiddo"}
modelOnlySites=${2:-True}

ntiles=44
#read in the tiles
rootdirMuSAruns="/kyukon/data/gent/vo/000/gvo00090/vsc44965/Doctoraat/Python/Machine_learning/Traditional_MLA/sd/MuSA/test/"

scriptdir="$(dirname "$0")"
jobscript="/kyukon/data/gent/vo/000/gvo00090/vsc44965/Doctoraat/Python/Machine_learning/Traditional_MLA/sd/MuSA/runMuSAtile.sh"


job=$(sbatch --job-name=test_MuSArun \
  --output=logs/output_test_MuSArun_%A.log \
  --error=logs/error_test_MuSArun_%A.log \
  --nodes=1 \
  --cpus-per-task=12 \
  --time="01:00:00" \
  --cluster=${cluster} \
  --array=0-$((ntiles-1)) \
  --wrap="$jobscript \${SLURM_ARRAY_TASK_ID} \
                        $rootdirMuSAruns \
                        $modelOnlySites")

#--array=0-$((ntiles-1)) \