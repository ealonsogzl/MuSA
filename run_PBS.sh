#!/bin/bash

# Simple example of running MuSA with a PBS job array

njobs=$1
nprocs=$2

# clean dirs
python clean.py

cat << end_jobarray > pbsScript.sh
#!/bin/bash
#PBS -N Musa
#PBS -J 1-${njobs}
#PBS -l select=1:ncpus=${nprocs}:mem=1000mb 
#PBS -l walltime=24:00:00

# Load software
module load gcc
module load conda
conda activate MuSA

# cd to directory from which qsub whas run 
cd "\${PBS_O_WORKDIR}"

# Run python script
python main.py "${njobs}" "${nprocs}" "\${PBS_ARRAY_INDEX}"

end_jobarray

qsub pbsScript.sh

rm pbsScript.sh
