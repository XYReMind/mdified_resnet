# Modified ResNet for ECE 6953 - Deep Learning 
By Xiaoyuan Lin 
Introduction: This model inspired the paper: https://arxiv.org/abs/1603.05027. It reaches 91.45% accuracy on the CIFAR-10 image classifaction dataset after 100 epochs and it contains ~1M parameters, which satisfied the project requirement. 

Total params: 1,711,322

Trainabale params: 1,711,322

Non-trainable params: 0 

Input size (MB): 0.01

Forward.backward pass size (MB): 87.88

Params size (MB): 6.53

Estimated Total Size (MB): 94.42

## Environment: NYU HPC Greene

A guide to using Greene can be found here: 
https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene/getting-started?authuser=0

You also need to connect to NYU VPN before using Greene. https://www.nyu.edu/life/information-technology/infrastructure/network-services/vpn.html is a guide to using VPN. 

RTX8000 NVIDIA GPU is used for this preject. Mem is set to 8GB. CPU-per-task is set to 1. node = 1

## Setup and run  

1. ssh into NYU green
2. Using the btach file provided in this repo. Run "sbatch min_project.sh"
3. Using command "squeue -u USERNAME" to check the progress of the training. 
4. Result will be store in file names "min_preoject.out"


