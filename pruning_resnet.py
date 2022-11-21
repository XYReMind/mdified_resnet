'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from time import perf_counter
import time
from torchsummary import summary

import utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import prune


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

args, unknown = parser.parse_known_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=4)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # uncomment these two lines for C7, without bactch norm layer.
        # and comment out the above two lines at the same time
        #out = F.relu(self.conv1(x))
        #out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # uncomment this line for C7, without bactch norm layer.
        # and comment out the above line at the same time
        #out = F.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


net = ResNet18()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4)

# for C6, uncomment the related optimizer.
#with nesterov
#optimizer = optim.SGD(net.parameters(), lr=args.lr, nesterov=True,momentum=0.9, weight_decay=5e-4)

#Adagrad
#optimizer = optim.Adagrad(net.parameters(), lr=args.lr, lr_decay=0, weight_decay=5e-4)

# Adadelta
#optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=5e-4)

# Adam
#optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Total number of trainable parameters: '+ str(params))

# Training
global_data_loader_time = 0
global_trainig_time = 0
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    best_acc = 0

    train_loss = 0
    correct = 0
    total = 0
    training_time_counter = 0
    data_loading_time_counter = 0

    total_start = perf_counter()
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        data_loading_time = time.time()-end

        inputs, targets = inputs.to(device), targets.to(device)

        training_time_start = perf_counter()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        training_time_stop = perf_counter()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        end = time.time()
        training_time_counter += (training_time_stop-training_time_start)
        data_loading_time_counter += data_loading_time
    global global_data_loader_time
    global_data_loader_time = data_loading_time_counter

    global global_trainig_time
    global_trainig_time = training_time_counter

    total_stop = perf_counter()
    print("Total running time for this epoch:",total_stop-total_start)
    print("Data-loading time for each epoch:",data_loading_time_counter)
    print("Train time for each epoch:",training_time_counter)
    epoch_loss = train_loss / len(trainloader)
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    print('Loss: %.5f | Best Acc: %.5f%%'  % (epoch_loss, best_acc))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def get_prune_model(self,pruning_method,ratio):

        '''
        Parameters :
        ------------
        model (object) : model that will be pruned
        pruning_method (str) : method of pruning that will be used
                               global     : see pytorch global_unstructured function
                               uniform    : see pytorch prune function, applied to each layer
        ratio (float) : ratio for pruning
        Returns :
        ---------
        model : pruned model with a weight_mask (not remove)
                For more information, see how pruning is done in pytorch and remove function
        '''

        if pruning_method == 'uniform':
            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) :
                    module = prune.l1_unstructured(module, 'weight', ratio)

        elif pruning_method == 'global':

            parameters_to_prune = []

            for name, module in self.named_modules():
                if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.AvgPool2d)  :
                    parameters_to_prune.append((module,'weight'))

            prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=ratio)

        return self

counter =0
training_counter = 0
for epoch in range(start_epoch, start_epoch+5):
    train(epoch)
    counter += global_data_loader_time
    #test(epoch)
    training_counter += global_trainig_time
    scheduler.step()


grads = []
for param in net.parameters():
    grads.append(param.grad.view(-1))
grads = torch.cat(grads)
print(grads.shape)

