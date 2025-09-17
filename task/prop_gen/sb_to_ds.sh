#!/bin/bash
#SBATCH -c 16
#SBATCH -t 20:00:00
#SBATCH -p bigmem,bigmem_intermediate
#SBATCH --mem=1000G
#SBATCH -o log.%j.out
#SBATCH -e log.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=pehlevan_lab

source ../../../venv_imply/bin/activate
python to_dataset.py