#!/bin/bash
#SBATCH -c 16
#SBATCH -t 3-00:00
# #SBATCH -p kempner_requeue
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-18%12
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --constraint="h100|h200"


source ../../../../../venv_imply/bin/activate
export WANDB_LOG_MODEL=false
# export WANDB_API_KEY=$(cat ~/wandb.txt)
# wandb login
python run.py ${SLURM_ARRAY_TASK_ID}


