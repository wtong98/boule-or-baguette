#!/bin/bash
#SBATCH -c 16
#SBATCH -t 12:00:00
# #SBATCH -p kempner_requeue
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o log_eval.%A.%a.out
#SBATCH -e log_eval.%A.%a.err
#SBATCH --array=1-18%12
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads
#SBATCH --constraint="h100|h200"


module load gcc
module load cuda/12.9

source /n/netscratch/pehlevan_lab/Lab/wlt/envs/venv_imply_uv_local/bin/activate
# export WANDB_API_KEY=$(cat ~/wandb.txt)
# wandb login
python eval.py ${SLURM_ARRAY_TASK_ID} 


