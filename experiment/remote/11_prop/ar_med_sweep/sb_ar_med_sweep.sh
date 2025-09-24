#!/bin/bash
#SBATCH -c 16
#SBATCH -t 3-00:00
#SBATCH -p kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=128000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-12
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads
# #SBATCH --exclude=holygpu8a19205,holygpu8a19503

source ../../../../../venv_imply/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}

