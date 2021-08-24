#!/bin/bash

#SBATCH --job-name=Training
#SBATCH --constraint=A100
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=micah.bowles@postgrad.manchester.ac.uk
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=17
#SBATCH --no-reque
#SBATCH --array=0-14%15

echo ">>> start"
### ARRAY JOBS ###
# The selection of `--array=1-5%4` means that we run jobs with
# array indexes 0-14 and use at most 15 nodes at once.
# We can pass the index using the variable: `$SLURM_ARRAY_TASK_ID`

echo ">>> training for fisher experiment"
echo ">>> training for fisher experiment"
# Manual array creation
CFGS=()
while IFS= read -r line; do
  [[ "$line" =~ ^#.*$ ]] && continue
  arr+=("$line")
  echo "$line"
done < configs/experiment_configs.txt

CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}

echo 'Training:' $CFG
python train.py --config $CFG
