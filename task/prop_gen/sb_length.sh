#!/bin/bash
#SBATCH -c 112
#SBATCH -t 2-00:00
#SBATCH --contiguous
#SBATCH -p sapphire
#SBATCH --mem=128000
#SBATCH -o log.%j.out
#SBATCH -e log.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=pehlevan_lab

source ../../../../../venv_imply/bin/activate
python generate.py
mv /scratch/data.json ~/scratch
