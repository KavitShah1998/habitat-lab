#!/bin/bash
#SBATCH --job-name=TEMPLATE_bc_train
#SBATCH --output=TEMPLATE_bc_train.out
#SBATCH --error=TEMPLATE_bc_train.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --partition short
#SBATCH --cpus-per-task=6
#SBATCH --exclude calculon,claptrap,alexa,bmo,cortana,oppy,fiona
##SBATCH -w jarvis

source ~/.bashrc

conda activate aug26n
cd HABITAT_REPO_PATH
python -u habitat_baselines/rl/behavioral_cloning/behavioral_cloning_v5.py \
 TEMPLATE \
 CONFIG_YAML
