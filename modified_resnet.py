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
import math

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

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

batch_size = 128
num_workers = 4
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

valloader = torch.utils.data.DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class Bottleneck(nn.Module):
    #expansion = 4

    def __init__(self, inplanes, outplanes, stride=1):

        assert outplanes % 4 == 0
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.bottleneck_planes = int(outplanes / 4)
        self.stride = stride
        self._make_layer()

    def _make_layer(self):
        # conv 1x1
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.conv1 = nn.Conv2d(self.inplanes, self.bottleneck_planes,kernel_size=1, stride=self.stride, bias=False)
        # conv 3x3
        self.bn2 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv2 = nn.Conv2d(self.bottleneck_planes, self.bottleneck_planes,kernel_size=3, stride=1, padding=1, bias=False)
        # conv 1x1
        self.bn3 = nn.BatchNorm2d(self.bottleneck_planes)
        self.conv3 = nn.Conv2d(self.bottleneck_planes, self.outplanes, kernel_size=1, stride=1)

        if self.inplanes != self.outplanes:
            self.shortcut = nn.Conv2d(self.inplanes, self.outplanes, kernel_size=1, stride=self.stride, bias=False)
        else:
            self.shortcut = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        # we do pre-activation
        out = self.relu(self.bn1(x))
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        out += residual
        return out


class ResNet(nn.Module):

    def __init__(self, block, output_classes=10):

        super(ResNet, self).__init__()
        n = 18
        nstages = [16, 64, 128, 256]
        # one conv at the beginning (spatial size: 32x32)
        self.conv1 = nn.Conv2d(3, nstages[0], kernel_size=3, stride=1,padding=1, bias=False)

        # use `block` as unit to construct res-net
        # Stage 0 (spatial size: 32x32)
        self.layer1 = self._make_layer(block, nstages[0], nstages[1], n, stride=1)
        # Stage 1 (spatial size: 32x32)
        self.layer2 = self._make_layer(block, nstages[1], nstages[2], n, stride=2)
        # Stage 2 (spatial size: 16x16)
        self.layer3 = self._make_layer(block, nstages[2], nstages[3], n, stride=2)
        # Stage 3 (spatial size: 8x8)
        self.bn = nn.BatchNorm2d(nstages[3])
        self.relu = nn.ReLU(inplace=True)
        # classifier
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(nstages[3], output_classes)

        # weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, inplanes, outplanes, nstage, stride=1):
        layers = []
        layers.append(block(inplanes, outplanes, stride))
        for i in range(1, nstage):
            layers.append(block(outplanes, outplanes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet_164(output_classes):
    model = ResNet(Bottleneck)
    return model


net = resnet_164(output_classes=10)

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
#optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=1e-4)

# uncomment the related optimizer.
#with nesterov
optimizer = optim.SGD(net.parameters(), lr=args.lr, nesterov=True,momentum=0.9, weight_decay=1e-4)

#Adagrad
#optimizer = optim.Adagrad(net.parameters(), lr=args.lr, lr_decay=0, weight_decay=5e-4)

# Adadelta
# optimizer = optim.Adadelta(net.parameters(), lr=args.lr, weight_decay=5e-4)

# Adam
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# For Q4 and Q5, print out the number of trainable parameters
params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Total number of trainable parameters: '+ str(params))


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    best_acc = 0
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = train_loss / len(trainloader)
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    print('Train Loss: %.5f | Train Best Acc: %.5f%%'  % (epoch_loss, best_acc))


def val(epoch):
    global best_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = val_loss / len(valloader)
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    print('Val Loss: %.5f | Val  Best Acc: %.5f%%'  % (epoch_loss, best_acc))

    # Save checkpoint.
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

    epoch_loss = test_loss / len(testloader)
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    print('Test Loss: %.5f | Test  Best Acc: %.5f%%'  % (epoch_loss, best_acc))

    # Save checkpoint.
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


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    val(epoch)
    test(epoch)
    scheduler.step()