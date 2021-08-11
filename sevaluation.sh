#!/bin/bash

#SBATCH --job-name=e2Training
#SBATCH --constraint=A100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --no-reque

echo ">>> start"

echo ">>> evaluation of fisher experiment"
python evaluate.py >& evaluate.log
