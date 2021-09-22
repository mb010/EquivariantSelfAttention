#!/bin/bash

#SBATCH --job-name=Evaluations
#SBATCH --constraint=A100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --no-reque
#SBATCH --array=0-1%2

echo ">>> start"
### ARRAY JOBS ###
# The selection of `--array=1-5%4` means that we run jobs with
# array indexes 0-14 and use at most 15 nodes at once.
# We can pass the index using the variable: `$SLURM_ARRAY_TASK_ID`

echo ">>> Evaluation"
# Manual array creation
CFGS=("5kernel_bowles2021_mirabest_RandAug.cfg" "5kernel_D8_attention_mirabest_RandAug.cfg")

CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}

echo '>>> Creating Figures:' $CFG
python -u utils/figures.py --config $CFG

echo '>>> Plotting figures for:' $CFG
python -u create_figures.py --config $CFG

echo '>>> Extracting data for:' $CFG
python -u extract_best.py --config $CFG
