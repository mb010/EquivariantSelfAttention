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

echo ">>> training for fisher experiment"
CFG=D16_attention_mirabest
echo $CFG
python train.py --config $CFG.cfg


#while read cfg; do
#    echo $cfg
#    python train.py --config $cfg.cfg
#done < fisher_configs.txt

#echo ">>> training on mingoLoTSS"
#python train.py --config bowles2021mingo-RandAug.cfg >& logs/bowles2021mingo-RandAug.log
