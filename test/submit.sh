#!/bin/bash
#SBATCH --job-name=test1
#SBATCH --partition=gpu

# ---- Resources per node ----
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # 1 MPI rank per GPU
#SBATCH --cpus-per-task=20           # 2 × 20 = 40 CPUs 
#SBATCH --gres=gpu:1                 # 2 GPUs per node

# ---- Memory (safe headroom) ----
#SBATCH --mem=100G                   # Node has ~181G total

# ---- Time & logging ----
#SBATCH --time=48:00:00
#SBATCH --output=log.out
#SBATCH --error=log.err

# ---- Safety & debugging ----
set -euo pipefail
set -x

# ---- Environment ----
module purge
module load mldl/cuda/cuda-12.9
module load nvhpc/25.7

cd "$SLURM_SUBMIT_DIR"

echo "Job started on $(date)"
echo "Nodes allocated: $SLURM_JOB_NODELIST"
echo "Total MPI tasks: $SLURM_NTASKS"
echo "CPUs per task:   $SLURM_CPUS_PER_TASK"

nvidia-smi
free -h

srun --gres=gpu:1 \
     stdbuf -oL -eL ../artracfd -m gpu -n 1*1*1

echo "Job finished on $(date)"

