# Modified ResNet for ECE 6953 - Deep Learning 


Introduction: This model inspired the paper: https://arxiv.org/abs/1603.05027. It reaches 91.45% accuracy on the CIFAR-10 image classifaction dataset after 100 epochs and it contains ~1M parameters, which satisfied the project requirement. 

![Alt text](/assets/arch.png?raw=true "ResNet164 Architecture")

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


## Experiment Progress(use only 50 epochs)

First try different optimizer

| Optimizer          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD               | 0.39477       | 86.52%           | 0.59443  | 83.48%      |
| SGD with Nesterov | 0.37395       | 87.2325%         | 0.61652  | 84.51%      |
| Adagrad           | 0.53728       | 81.36%           | 0.84589  | 77.07%      |
| Adadelta          | 0.29821       | 89.735%          | 0.59151  | 81.84%      |
| Adam              | 1.92809       | 24.845%          | 1.91847  | 28.04%      |

SGD and SGD with Nesterov perform better.

Then use these two optimzier for the following parameters tuning. 

1. Batch size 64 with num workers = 2, lr=0.1, momentum=0.9, weight_deca=5e4

| Optimizer          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD               | 0.58887       | 79.83%           | 1.13847  | 79.03%      |
| SGD with Nesterov | 0.56224       | 80.6625%         | 0.67857  | 80.16%      |

2. Use only SGD with Nesterov since it performs better than SGD in both cases. Change the num workers. Num worker = 4 batch size = 128, lr=0.1, momentum=0.9, weight_deca=5e4

| Optimizer          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD with Nesterov | 0.3983        | 86.2475%         | 0.55617  | 83.01%      |

3. Num worker = 2 batch size =256, lr=0.1, momentum=0.9, weight_deca=5e4

| Optimizer          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD with Nesterov | 0.27637       | 90.53%           | 0.61719  | 83.73%      |

4. Num worker = 2 batch size =128, lr=0.1, momentum=0.9, weight_deca=1e4

| Optimzier          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD with Nesterov | 0.19765       | 93.055%          | 0.35552  | 88.32%      |

5. Num worker = 4 batch size =128, lr=0.1, momentum=0.9, weight_deca=1e4 *(BEST)

| Optimizer          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD with Nesterov | 0.19466       | 93.17%           | 0.37038  | 88.88%      |

6. Num worker = 4 batch size =256, lr=0.1, momentum=0.9, weight_deca=1e4

| Optimizer          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD with Nesterov | 0.13924       | 95.035%          | 0.52717  | 87.54%      |

7. Num worker = 6 batch size =128, lr=0.1, momentum=0.9, weight_deca=1e4

| Optimzier          | Training loss | Train Best Acc   |Val loss  |Val Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|------------:|
| SGD with Nesterov | 0.19155       | 93.2775%         | 0.37363  | 88.25%      |


## Result 

Num worker = 4 batch size =128, lr=0.1, momentum=0.9, weight_deca=1e4
After 100 epoch 

| Optimizer          | Training loss | Train Best Acc   |Test loss |Test Best Acc |
| ----------------- |:-------------:| ----------------:|---------:|-------------:|
| SGD with Nesterov | 0.08309       | 97.135%          | 0.35683  | 91.45%       |


Reference: 

https://github.com/KaimingHe/resnet-1k-layers/blob/master/README.md

[1] Canatalay, Peren & UÃ§an, Osman & Zontul, Metin. (2021). DIAGNOSIS OF BREAST CANCER FROM X-RAY IMAGES USING DEEP LEARNING METHODS. PONTE International Scientific Researches Journal. 77. 10.21506/j.ponte.2021.6.1. 
