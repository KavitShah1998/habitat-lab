#!/bin/bash
#SBATCH --job-name=TEMPLATE_bc_eval
#SBATCH --output=TEMPLATE_bc_eval.out
#SBATCH --error=TEMPLATE_bc_eval.err
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --partition short
#SBATCH --cpus-per-task=6
#SBATCH --exclude calculon,claptrap,alexa,bmo,cortana,oppy,fiona
##SBATCH -w jarvis

source ~/.bashrc

conda activate aug26n
cd HABITAT_REPO_PATH
python -u habitat_baselines/rl/behavioral_cloning/evaluate_student.py \
 TEMPLATE \
 CONFIG_YAML
