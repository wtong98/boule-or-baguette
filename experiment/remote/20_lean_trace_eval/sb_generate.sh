#!/bin/bash
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=0-3
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads
#SBATCH --constraint="h100|h200"

set -euo pipefail

module load gcc
module load cuda/12.9

VLLM_ENV="${VLLM_ENV:-/n/netscratch/pehlevan_lab/Lab/wlt/envs/venv_imply_vllm}"
source "${VLLM_ENV}/bin/activate"

python generate.py --task-index "${SLURM_ARRAY_TASK_ID}"
