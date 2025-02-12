#!/bin/bash
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p seas_compute,shared
#SBATCH --mem=128000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-60%15
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=pehlevan_lab
#SBATCH --exclude=holygpu8a19604

source ../../../../../venv_haystack/bin/activate
python run.py ${SLURM_ARRAY_TASK_ID}

