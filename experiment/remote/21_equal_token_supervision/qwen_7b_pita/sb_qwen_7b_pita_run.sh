#!/bin/bash
#SBATCH -c 16
#SBATCH -t 3-00:00
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=0-39
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads


source ../../../../../venv_imply/bin/activate
export WANDB_LOG_MODEL=false
python config.py "${SLURM_ARRAY_TASK_ID}" run
