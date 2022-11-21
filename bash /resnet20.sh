#!/bin/bash
#SBATCH --job-name=min_project
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

module purge
module load anaconda3/2020.07
module load python/intel/3.8.6
pip install torchsummary

#pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

cd /scratch/xl3914/dl
python3 resnet20.py

