#!/bin/bash
#SBATCH -c 16
#SBATCH -t 3-00:00:00
#SBATCH -p kempner_requeue
#SBATCH --gres=gpu:1
#SBATCH --mem=96000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-12
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=kempner_pehlevan_lab
#SBATCH --exclude=holygpu8a19205,holygpu8a19503

source ../../../../../venv_imply/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}

