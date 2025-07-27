#!/bin/bash
#SBATCH -c 8
#SBATCH -t 1-06:00:00
#SBATCH -p kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=64000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-36%12
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_grads
#SBATCH --exclude=holygpu8a19505,holygpu8a19105,holygpu8a19301

source ../../../../../venv_imply/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}

