#!/bin/bash

#SBATCH --job-name=Evaluations
#SBATCH --constraint=A100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --no-reque
#SBATCH --array=0-23%24

echo ">>> start"
### ARRAY JOBS ###
# The selection of `--array=1-5%4` means that we run jobs with
# array indexes 0-14 and use at most 15 nodes at once.
# We can pass the index using the variable: `$SLURM_ARRAY_TASK_ID`

echo ">>> Extracting Essentials"
# Manual array creation
CFGS=()
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ ]] && continue
  CFGS+=("$line")
done < configs/experiment_configs.txt

CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}

echo 'Extracting data for:' $CFG
python extract_best.py --config $CFG
