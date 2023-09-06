#!/bin/bash

# Simple example of running MuSA with a PBS job array

njobs=$1
nprocs=$2

declare -i mempcpu=2000
units="mb"

memory=$((mempcpu * nprocs))
memorystrg=$memory$units

# clean dirs
python clean.py

cat << end_jobarray > pbsScript.sh
 
#!/bin/bash
#SBATCH --job-name=Musa
#SBATCH --array=1-${njobs}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=${nprocs}
#SBATCH --mem-per-cpu=${memorystrg}
#SBATCH --time=24:00:00

# Load software
module load gcc
module load conda
conda activate MuSA

# cd to directory from which qsub whas run 
cd "\${SLURM_SUBMIT_DIR}"

# Run python script
python main.py "${njobs}" "${nprocs}"

end_jobarray

sbatch slurmScript.sh

rm slurmScript.sh
