#!/bin/bash

#SBATCH --job-name=Evaluations
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
CFGS=(
  # Model Testing
  '5kernel_bowles2021_mirabest_RandAug.cfg'
  '5kernel_scaife2021_mirabest_RandAug.cfg'
  '5kernel_D16_scaife2021_mirabest_RandAug.cfg'
  '5kernel_C4_attention_mirabest_RandAug.cfg'
  '5kernel_C8_attention_mirabest_RandAug.cfg'
  '5kernel_C16_attention_mirabest_RandAug.cfg'
  '5kernel_D4_attention_mirabest_RandAug.cfg'
  '5kernel_D8_attention_mirabest_RandAug.cfg'
  '5kernel_D16_attention_mirabest_RandAug.cfg'
  # Kernel Testing
  #'5kernel_bowles2021_mirabest-RandAug.cfg' # Already trained in 'Model Testing'
  #'5kernel_scaife2021_mirabest_RandAug.cfg' # Already trained in 'Model Testing'
  #'5kernel_D8_attention_mirabest-RandAug.cfg' # Already trained in 'Model Testing'
  '7kernel_bowles2021_mirabest_RandAug.cfg'
  '7kernel_scaife2021_mirabest_RandAug.cfg'
  '7kernel_D8_attention_mirabest_RandAug.cfg'
  '9kernel_bowles2021_mirabest_RandAug.cfg'
  '9kernel_scaife2021_mirabest_RandAug.cfg'
  '9kernel_D8_attention_mirabest_RandAug.cfg'
)
CFG=${CFGS[$SLURM_ARRAY_TASK_ID]}

echo 'Extracting data for:' $CFG
python extract_best.py --config $CFG
