#!/bin/bash
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p pehlevan,bigmem,bigmem_intermediate,seas_compute,sapphire,shared,
#SBATCH --mem=128000
#SBATCH -o log.%j.out
#SBATCH -e log.%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=wtong@g.harvard.edu
#SBATCH --account=pehlevan_lab

source ../../../venv_imply/bin/activate
python to_dataset.py