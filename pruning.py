import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision import models
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms
from torch.nn.utils import prune
##dataset
from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
##libraries
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
##scripts
import resnet
import binaryconnect
import utils



#### check GPU usage ####

use_gpu = torch.cuda.is_available()
if use_gpu:
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

#### add parser ####

parser = argparse.ArgumentParser()

## infos
parser.add_argument('--name', type = str , default = 'demo', help = 'name of the experience' )
parser.add_argument('--score', action='store_true' , default = False, help = 'micronet score')

## model used
parser.add_argument('--modelToUse', type = str, default = 'ResNet18' , choices = ['ResNet18','ResNet34'], help ='Choose ResNet model to use')
parser.add_argument("--num_blocks", type=int, nargs="+", default=[2, 2, 2, 0])
parser.add_argument("--power_in_planes",type = int, default=4)
## dataset



parser.add_argument('--dataset', type = str , choices = ['minicifar','cifar10','cifar100'] , default = 'minicifar' )

## training settings
parser.add_argument('--train', action='store_true' , default = False, help = 'perform training')
parser.add_argument('--ptrain', action='store_true' , default = False, help = 'perform iterative/pruning training')
parser.add_argument('--lr', type = float, default = 1e-2 , help = 'Learning rate')
parser.add_argument('--momentum', type = float, default = 0.9 , help = 'momentum for Learning Rate')
parser.add_argument('--decay', type = float, default = 5e-4 , help = 'decay')
parser.add_argument('--epochs', type = int, default = 300  , help = 'Number of epochs for training')
parser.add_argument('--batch_size', type = int, default = 32 , help ='Batch size for DataLoader')
parser.add_argument('--overfitting', type = str , default = 'loss' , choices = ['loss','accuracy'], help ='Choose overfitting type')
parser.add_argument('--optimizer', type = str , choices = ['sgd','adam'], default = 'sgd' )

parser.add_argument('--test', action='store_true' , default = False, help = 'perform test')
parser.add_argument('--path', type = str , help = 'path to pth in desired logs to find model_weights')

## quantization
parser.add_argument('--pruning', action='store_true' , default = False, help = 'perform pruning' )
parser.add_argument('--method',  type = str , choices = ['uniform','global','decreasing'], default = 'global' )
parser.add_argument('--bin', action='store_true' , default = False, help = 'perform binarization' )

parser.add_argument('--ratio', type = float, default = 0.3 , help = 'ratio for pruning')

args = parser.parse_args()


#### choose dataset and set dataloaders ####

def get_model_dataset(dataset,batch_size,modelToUse):
    '''
    Parameters :
    ------------
    dataset (str) : name of the dataset to use
    batch_size (int) : size of the batchs for dataloaders
    modelToUse (str) : name of the model to used
    Returns :
    ---------
    model : wanted model
    train/valid/testloaders : dataloaders for training, validation and test
    '''
    if dataset == 'minicifar':

        trainloader = DataLoader(minicifar_train,batch_size=batch_size,sampler=train_sampler)
        validloader = DataLoader(minicifar_train,batch_size=batch_size,sampler=valid_sampler)
        testloader = DataLoader(minicifar_test,batch_size=batch_size)
        n = 4

    elif dataset == 'cifar10':

        ## add data augmentation
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        dataset = CIFAR10(root='data/', download=True, transform=transform_train)
        test_dataset = CIFAR10(root='data/', train=False, transform=transform_test)
        val_size = 5000
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        trainloader = DataLoader(dataset,batch_size=batch_size)
        validloader = DataLoader(test_dataset,batch_size=batch_size)
        testloader = DataLoader(test_dataset,batch_size=batch_size)
        n = 10

    elif dataset == 'cifar100':

        transform_train_1 = [transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]

        transform_test_1 = [transforms.ToTensor(),
                     transforms.Normalize(mean=[n/255.
                        for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])]

        transform_train = transforms.Compose(transform_train_1)
        transform_test = transforms.Compose(transform_test_1)

        dataset = CIFAR100(root='data/', download=True, transform=transform_train)
        test_dataset = CIFAR100(root='data/', train=False, transform=transform_test)

        val_size = 10000
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
        trainloader = DataLoader(dataset,batch_size=batch_size)
        validloader = DataLoader(test_dataset,batch_size=batch_size)
        testloader = DataLoader(test_dataset,batch_size=batch_size)
        n = 100


    if modelToUse == 'ResNet18' :
        model = resnet.ResNet18(N=n, num_blocks = args.num_blocks, power_in_planes = args.power_in_planes)

    elif modelToUse == 'ResNet34' :
        model = resnet.ResNet34(N=n, num_blocks = args.num_blocks, power_in_planes = args.power_in_planes)

    return model , trainloader , validloader , testloader

def get_sparsity(model):
    '''
    Parameters :
    ------------
    model (object) : model that will be pruned
    Prints :
    ---------
    Percentage of zeros in each layer
    '''
    L = []
    for name1, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.AvgPool2d)  :
            L.append((module,'weight'))
            txt1 = "Sparsity in {}t: {:.2f}%".format(name1,
                100. * float(torch.sum(module.weight == 0))
                / float(module.weight.nelement()))
            print(txt1)

    sum = 0
    totals = 0
    for tuple in L :
        sum += torch.sum(tuple[0].weight == 0)
        totals += tuple[0].weight.nelement()
        txt = "Global sparsity: {:.2f}%".format(sum/totals * 100)
    print(txt)

def pos_zeros(model):
    '''
    Parameters :
    ------------
    model (object) : model that will be pruned
    Prints :
    ---------
    Number of feature maps that have more than 50% of zeros for each layer
    '''
    L = []

    for name1, module in model.named_modules():
        zeros = []
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            L.append((module,'weight'))
            for i in range(module.weight_mask.shape[0]):
                A = int(100*float(torch.sum(module.weight_mask[i] == 0))/float(module.weight_mask[i].nelement()))
                zeros.append(A)
            print('{0:20} {1} / {2}'.format(name1, len([x for x in zeros if x > 50.0]),len(zeros)))



def get_nb_params(model):
    return sum(p.numel() for p in model.parameters())

## get model and dataloaders

backbonemodel , trainloader , validloader , testloader = get_model_dataset(args.dataset,args.batch_size,args.modelToUse)

##### check number of parameters ####
params = get_nb_params(backbonemodel)

#### print and save experince config ####
print('='*10 + ' EXPERIENCE CONFIG ' + '='*10)
print('{0:20} {1}'.format('model', args.modelToUse))
print('{0:20} {1}'.format('Nb of parameters',params))

for arg in vars(args):
    print('{0:20} {1}'.format(arg, getattr(args, arg)))
print('{0:20} {1}'.format('GPU',use_gpu))
print('='*10 + '==================' + '='*10)


#### create optimizer, criterion and scheduler ####

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(backbonemodel.parameters(), lr=args.lr, momentum=args.momentum,weight_decay=args.decay)
#optimizer = optim.Adam(backbonemodel.parameters(), lr=args.lr,weight_decay=args.decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

## binarization of the model
if args.bin:
    backbonemodel = binaryconnect.BC(backbonemodel)

## if pruning is selected, then prune model and print its sparsity
if args.pruning:
    backbonemodel = backbonemodel.get_prune_model(args.method,args.ratio)
    get_sparsity(backbonemodel)


if args.train :
    ## load model in the device (cpu or cuda)
    backbonemodel = backbonemodel.to(device)

    ## train function
    backbonemodel.train_model(trainloader,validloader,criterion,optimizer,scheduler,args.overfitting,args.epochs,args.name)

    ## save config in a text file
    f = open('./logs/{}/experience_config.txt'.format(args.name),'w+')
    f.write('='*10 + ' EXPERIENCE CONFIG ' + '='*10)
    f.write('\n')
    for arg in vars(args):
        f.write('{0:20} {1}'.format(arg, getattr(args, arg)))
        f.write('\n')
    f.write('{0:20} {1}'.format('GPU',use_gpu))
    f.write('\n')
    f.write('{0:20} {1}'.format('Nb of parameters',params))
    f.write('\n')
    f.write('='*10 + '==================' + '='*10)
    f.close()

### ptrain is training and pruning iteratively
elif args.ptrain :

    ## load the model in the gpu or cpu
    backbonemodel = backbonemodel.to(device)

    ## first validation to see the first model
    val_loss,val_acc = backbonemodel.validate(validloader,criterion,0)
    print(' == First validation before training == ' )
    print('  -> Validation Loss     = {}'.format(val_loss))
    print('  -> Validation Accuracy = {}'.format(val_acc))

    ## training and pruning processes
    ratio = args.ratio
    for i in range(3):
        print(' = '*10 )

        ## increase pruning ratio
        dratio = 0.20
        ratio += (1-ratio)*dratio

        ## prune the model
        backbonemodel = get_prune_model(backbonemodel,args.method, dratio)

        backbonemodel.train_model(trainloader,validloader,criterion,optimizer,args.epochs,args.name+ str(ratio))

## test process
if args.test :
    ## .half to divide by 2 the precision on weights
    backbonemodel = backbonemodel.to(device)
    ##test process
    test_loss, test_acc = backbonemodel.test(testloader,criterion,device)
    utils.save_test_results(args.path,test_acc,test_loss,args.pruning,args.ratio)

else:
    sys.exit('Need to select either --train or --test or --ptrain')