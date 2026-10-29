#!/bin/bash
# Runs one QPE job: python qpe_job.py ID (see qpe_job.py for what each ID runs).
#
# Submit from demo/ as PBS array jobs (the array index is the job ID):
#   GPU, all error budgets (4 ... 14 phase qubits):  qsub -J 0-10 run_qpe.sh
#   CPU, error budgets up to 10 phase qubits:        qsub -J 11-17 run_qpe.sh
# Or run a single job directly:                      bash run_qpe.sh 3
#
#PBS -N fesh4_qpe
#PBS -l instance_type=g4dn.2xlarge
#PBS -l walltime=72:00:00
#PBS -l mem=30gb
#PBS -j oe

# Job ID: the first argument, or the array index when PBS runs it.
ID=${1:-$PBS_ARRAY_INDEX}

# Run from demo/: the folder the job was submitted from.
cd "${PBS_O_WORKDIR:-$(pwd)}"

# Activate the virtual environment.
source ~/.venvs/quiche-cudaq/bin/activate

# Threads for CUDA-Q's CPU backend: one per physical core (half the logical CPUs).
export OMP_NUM_THREADS=$(( $(nproc) / 2 ))

python qpe_job.py "$ID"
