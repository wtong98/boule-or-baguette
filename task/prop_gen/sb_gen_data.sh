#!/bin/bash
#SBATCH -c 16
#SBATCH -t 3-00:00
#SBATCH -p pehlevan,seas_compute,sapphire,shared,
#SBATCH --mem=64000
#SBATCH -o log.%A.%a.out
#SBATCH -e log.%A.%a.err
#SBATCH --array=1-24
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=pehlevan_lab

source ../../../venv_imply/bin/activate
python generate.py ${SLURM_ARRAY_TASK_ID}