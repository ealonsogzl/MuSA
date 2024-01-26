#!/bin/bash

# Simple example of running MuSA with a PBS job array

njobs=$1
nprocs=$2

# clean dirs
python clean.py

cat << end_jobarray > slurmScript.sh
#!/bin/bash
#SBATCH --export=none
#SBATCH --account=cesbio
#SBATCH --job-name=Musa
#SBATCH --array=1-${njobs}
#SBATCH -N 1
#SBATCH -n ${nprocs}
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00

# Load software
module load gcc
module load conda
conda activate MuSA

# cd to directory from which qsub whas run 
cd "\${SLURM_SUBMIT_DIR}"

# Run python script
python main.py "${njobs}" "${nprocs}" "\${SLURM_ARRAY_TASK_ID}"

end_jobarray

sbatch slurmScript.sh

rm slurmScript.sh
