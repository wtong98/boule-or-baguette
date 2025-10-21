#!/bin/bash
#SBATCH -c 16
#SBATCH -t 1-00:00
#SBATCH -p kempner,kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=4,6,8
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads


source ../../../../../venv_imply/bin/activate
# export WANDB_API_KEY=$(cat ~/wandb.txt)
# wandb login
python run.py ${SLURM_ARRAY_TASK_ID} imply


