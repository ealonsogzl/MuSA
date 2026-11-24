#!/bin/bash
#
# Run MuSA as a Slurm job array.
#
# Usage (from any directory):
#   bash run_slurm.sh <njobs> <nprocs>
#
#   njobs   number of array tasks. The cells of the domain are split among them
#   nprocs  CPUs (multiprocessing workers) used by each task, on a single node
#
# config.py must set parallelization = "HPC.array", MPI = False and
# implementation = "distributed" or "Spatial_propagation".
#
# Optional environment variables:
#   MUSA_ENV          conda environment (default: MuSA)
#   MUSA_MEM_PER_CPU  memory per CPU in GB (default: 1)
#   MUSA_WALLTIME     time limit of each task (default: 24:00:00)
#   CONDA_BASE        conda installation, if conda is not in PATH
#
# Add any option your cluster needs (partition, account...) to the #SBATCH
# lines below.

set -eo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[1-9][0-9]*$ || ! $2 =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash $0 <njobs> <nprocs>" >&2
    exit 1
fi
njobs=$1
nprocs=$2

env_name=${MUSA_ENV:-MuSA}
mem_per_cpu=${MUSA_MEM_PER_CPU:-1}
walltime=${MUSA_WALLTIME:-24:00:00}
musa_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
conda_base=${CONDA_BASE:-$("${CONDA_EXE:-conda}" info --base 2>/dev/null || true)}

if [[ ! -f ${conda_base}/etc/profile.d/conda.sh ]]; then
    echo "conda not found: activate conda first or set CONDA_BASE" >&2
    exit 1
fi

# Check config.py and compile FSM2 now, on the submission node, so that a
# wrong configuration or a compilation error is reported before queueing.
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${env_name}"
cd "${musa_dir}"
python - <<'EOF'
import os
import sys
import config as cfg

errors = []
if cfg.parallelization != "HPC.array":
    errors.append('parallelization = "HPC.array"')
if cfg.implementation not in ("distributed", "Spatial_propagation"):
    errors.append('implementation = "distributed" or "Spatial_propagation"')
if cfg.MPI:
    errors.append("MPI = False (each array task uses multiprocessing)")
if errors:
    sys.exit("config.py must set: " + "; ".join(errors))

if cfg.implementation == "Spatial_propagation":
    print(
        "WARNING: Spatial_propagation needs the prior maps generated in "
        "advance (pre_main_spatial.py) and all the array tasks running at "
        "the same time.",
        file=sys.stderr,
    )

if cfg.numerical_model == "FSM2":
    import modules.fsm_tools as model

    model.model_compile()
    if not os.path.isfile(model.model_binary_path()):
        sys.exit("FSM2 compilation failed")
EOF

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=MuSA
#SBATCH --array=1-${njobs}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${nprocs}
#SBATCH --mem-per-cpu=${mem_per_cpu}G
#SBATCH --time=${walltime}
#SBATCH --export=NONE

set -eo pipefail

# Batch shells are not interactive: load conda before activating
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${env_name}"

cd "${musa_dir}"
python main.py ${njobs} ${nprocs} "\${SLURM_ARRAY_TASK_ID}"
EOF
