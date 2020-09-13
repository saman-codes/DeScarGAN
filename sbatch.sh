#! /bin/bash

#SBATCH --job-name='ಠ_ಠ'
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output out/job%J.out
#SBATCH --error err/job%J.err
#SBATCH --partition=arti
#SBATCH --gres=gpu:1

module load cuda/10.0 

eval "$(conda shell.bash hook)"
conda activate descar_gan
python main.py
