#!/bin/bash
#SBATCH -c 16
#SBATCH -t 6:00:00
#SBATCH -p kempner_h100
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=0
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_pehlevan_lab


module load gcc
module load cuda/12.9

source /n/netscratch/pehlevan_lab/Lab/wlt/envs/venv_imply_uv_local/bin/activate
# export WANDB_API_KEY=$(cat ~/wandb.txt)
# wandb login
python eval.py


