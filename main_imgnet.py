'''Train CIFAR100 with PyTorch.'''
from __future__ import print_function
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
from time import sleep
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.datasets as datasets
import torch.utils.data as data

import copy
import torchvision
import torchvision.transforms as transforms
import math
import os
import argparse
# from torchviz import make_dot
# from IPython import embed
import sys
# cwd = os.getcwd()
cwd = '/home/pytorch-cifar/'
model_dir = os.path.join(cwd, 'models')
sys.path.append(model_dir)
from models import *
from utils import progress_bar
from tensorboardX import SummaryWriter
from random import randint
from PCANorm import *
from ZCANorm import *
import numpy as np


def isnan(x):
    return x != x

# torch.backends.cudnn.deterministic = True
# torch.manual_seed(999)
# torch.cuda.manual_seed_all(999)

parser = argparse.ArgumentParser(description='PyTorch Tiny ImgNet Training')
parser.add_argument('--norm', default='batchnorm', type=str, help='norm layer type')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

BatchSize = args.batch_size

# trainset = torchvision.datasets.CIFAR100(root='/tmp', train=True, download=True, transform=transform_train)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BatchSize, shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR100(root='/tmp', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=BatchSize, shuffle=False, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data_dir = '/home/pytorch-tiny-imagenet/tiny-imagenet-200/'
num_workers = {'train': 4, 'val': 0, 'test': 0}
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(20),  # transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
}
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val','test']}
dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=100, shuffle=True, num_workers=num_workers[x])
                  for x in ['train', 'val', 'test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}

trainloader = dataloaders['train']
testloader = dataloaders['test']
valloader = dataloaders['val']

# Model
norm = args.norm
norm = 'zcanormpiv2'  # 'batchnorm'  # 'zcanormpi'  # 'zcanormpiv2'
print('==> Building model using {}..'.format(norm))
if norm == 'batchnorm':
    Norm = nn.BatchNorm2d
elif norm == 'mybatchnorm':
    Norm = myBatchNorm
elif norm == 'groupnorm':
    Norm = nn.GroupNorm
elif norm == 'mygroupnorm':
    Norm = myGroupNorm  # myGroupNorm
elif norm == 'instancenorm':
    Norm = nn.InstanceNorm2d
elif norm == 'layernorm':
    Norm = nn.LayerNorm
elif norm == 'zcanorm':
    Norm = myZCANorm
elif norm == 'zcanormfloat':
    Norm = myZCANormfloat
elif norm == 'zcanormorg':
    Norm = ZCANormOrg
elif norm == 'zcanormpi':
    Norm = ZCANormPI
elif norm == 'zcanormpiunstable':
    Norm = ZCANormPIunstable
elif norm == 'zcanormpidebug':
    Norm = ZCANormPI_debug
elif norm == 'pcanorm':
    Norm = myPCANorm
elif norm == 'pcanormfloat':
    Norm = myPCANormfloat
elif norm == 'pcanorm-norec':
    Norm = myPCANorm_noRec
elif norm == 'zcanormpiv2':
    Norm = ZCANormPIv2
# net = VGG('VGG19')
# net = resnet18(Norm=Norm, num_classes=200)  # resnet18(Norm=Norm)
net = resnet50(Norm=Norm, num_classes=200)  # resnet50(Norm=Norm)
# net = resnet18nips(Norm=Norm, num_classes=10)
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)

save_dir = '/home/pytorch-cifar/pamitinyimgnet-pi'  # 'iclrcifar100'  # 'runs'
model_name = net._get_name()
id = randint(0, 1000)
# logdir = os.path.join(save_dir,'ResNet50group2/zcanormpiv2-bs128/736')
# logdir = os.path.join(save_dir, model_name+'18'+'group4e-10', '{}-bs{}'.format(norm, BatchSize), str(id))
logdir = os.path.join(save_dir, model_name+'50'+'group1e-2taylor9', '{}-bs{}'.format(norm, BatchSize), str(id))
# logdir = os.path.join(save_dir, model_name+'18'+'group1', '{}-bs{}rank16'.format(norm, BatchSize), str(id))
# _1layer95pct29pi _zcapi4group100pct19pi, _zcapi8group100pct19pi-debug,  _zcapi1group
if not os.path.isdir(logdir):
    os.makedirs(logdir)

writer = SummaryWriter(log_dir=logdir)
print('RUNDIR: {}'.format(logdir))

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(logdir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('{}/best-{}-ckpt.t7'.format(logdir, model_name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    scheduler.load_state_dict(checkpoint['scheduler'])
    optimizer.load_state_dict(checkpoint['optimizer'])

# Training
# net.load_state_dict(torch.load('net-31-1188.pt'))


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # loss_tm1 = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # for n, p in net.named_parameters():
        #     if p.requires_grad:
        #         if (p.data != p.data).any():
        #             print('net param {} weights contains NaN..'.format(n))
        #             print(p.data)
        #             while True:
        #                 sleep(1)

        optimizer.zero_grad()
        # inputs = torch.load('input-31-1188.pt')
        # targets = torch.load('target-31-1188.pt')
        # start_time = time.time()

        # outputs = net(inputs)

        # elapsed_time = time.time() - start_time
        # print('forward consumes time: {}'.format(elapsed_time))
        # embed()
        # dot = make_dot(outputs, params=dict(net.named_parameters()))
        # dot.render(str(batch_idx))

        # loss = criterion(outputs, targets)

        outputs = net(inputs)
        loss = criterion(outputs, targets)

        if isnan(loss):
            print('nan found')
            print('paras from previous update:')
            for n, p in net.named_parameters():
                if p.requires_grad and ("layer1" not in n) and ("layer2" not in n) and ("layer3" not in n) and (
                        "layer4" not in n):
                    print('param/{}mean'.format(n), p.abs().mean().item(), epoch * len(trainloader) + batch_idx + 1)
                    print('param/{}max'.format(n), p.abs().max().item(), epoch * len(trainloader) + batch_idx + 1)
            sys.exit("Error message")

        loss.backward()

        # try:
        #     outputs = net(inputs)
        #     loss = criterion(outputs, targets)
        #     loss.backward()
        # except:
        #     print("hi")
        #     # if hasattr(torch.cuda, 'empty_cache'):
        #     #     loss = None
        #     #     torch.cuda.empty_cache()
        #     continue

        # print('loss {}'.format(loss))

        # if math.isnan(loss):
        #     print('loss contains nan')
        #     print('checking if input contains nan..')
        #     if (inputs != inputs).any():
        #         print('input contains nan')
        #     else:
        #         print('input has no nan')
        #
        #     print('checking if input contains all 0s..')
        #     if inputs.sum().item() == 0:
        #         print('input is all 0s')
        #     else:
        #         print('input has no 0')
        #
        #     print('saving inputs targets net optimizer..')
        #     torch.save(inputs, 'input-{}-{}.pt'.format(epoch, batch_idx))
        #     torch.save(targets, 'target-{}-{}.pt'.format(epoch, batch_idx))
        #     torch.save(net.state_dict(), 'net-{}-{}.pt'.format(epoch, batch_idx))
        #     torch.save(optimizer.state_dict(), 'optimizer-{}-{}.pt'.format(epoch, batch_idx))
        #     while True:
        #         sleep(1)
        # print('backwarding..')

        # loss.backward()

        # loss_diff = np.abs(loss_tm1 - loss.item())

        # d = dict(net.named_buffers())
        # for key, value in d.items():
        #     print(key, value.shape)
        # for n, p in net.named_parameters():
        #     print('name: {}, shape: {}'.format(n, p.shape))

        # writer.add_scalar('grad/loss', loss.item(), epoch * len(trainloader) + batch_idx + 1)
        # for n, p in net.named_parameters():
        #     if p.requires_grad and("layer1" not in n) and ("layer2" not in n) and ("layer3" not in n) and ("layer4" not in n):
        #         writer.add_scalar('grad/{}mean'.format(n), p.grad.abs().mean().item(), epoch * len(trainloader) + batch_idx + 1)
        #         writer.add_scalar('grad/{}max'.format(n), p.grad.abs().max().item(), epoch * len(trainloader) + batch_idx + 1)

        # loss_tm1 = loss.item()
        # for n, p in net.named_parameters():
        #     if p.requires_grad:
        #         if (p.grad != p.grad).any():
        #             print('net {} param gradient contains NaN..'.format(n))
        #             print('loss is {}'.format(loss))
        #             print('saving inputs targets net optimizer..')
        #             torch.save(inputs, 'grad-input-{}-{}.pt'.format(epoch, batch_idx))
        #             torch.save(targets, 'grad-target-{}-{}.pt'.format(epoch, batch_idx))
        #             torch.save(net.state_dict(), 'grad-net-{}-{}.pt'.format(epoch, batch_idx))
        #             torch.save(optimizer.state_dict(), 'grad-optimizer-{}-{}.pt'.format(epoch, batch_idx))
        #             while True:
        #                 sleep(1)
        #         else:
        #             print('net {} param gradient is OK'.format(n))
        #         if (p.data != p.data).any():
        #             print('net {} param weights contains NaN..'.format(n))
        #             print('loss is {}'.format(loss))
        #             print('saving inputs targets net optimizer..')
        #             torch.save(inputs, 'weight-input-{}-{}.pt'.format(epoch, batch_idx))
        #             torch.save(targets, 'weight-target-{}-{}.pt'.format(epoch, batch_idx))
        #             torch.save(net.state_dict(), 'weight-net-{}-{}.pt'.format(epoch, batch_idx))
        #             torch.save(optimizer.state_dict(), 'weight-optimizer-{}-{}.pt'.format(epoch, batch_idx))
        #             while True:
        #                 sleep(1)
        #         else:
        #             print('net {} param weights is OK'.format(n))

        optimizer.step()
        if batch_idx % 500 == 0:
            writer.add_scalar('loss/train_loss', loss.item(), epoch*len(trainloader)+batch_idx+1)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100. * correct / total
    writer.add_scalar('loss/train_loss_avg', train_loss / len(trainloader), epoch)
    writer.add_scalar('train/accuracy', acc, epoch)
    writer.add_scalar('train/error', 100 - acc, epoch)


def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('loss/val_loss_avg', test_loss / len(valloader), epoch)
    writer.add_scalar('val/accuracy', acc, epoch)
    writer.add_scalar('val/error', 100-acc, epoch)
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        # if not os.path.isdir('checkpoint'):
        #     os.mkdir('checkpoint')
        # torch.save(state, os.path.join(logdir, 'best-{}-ckpt.t7'.format(model_name)))
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

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('loss/test_loss_avg', test_loss / len(testloader), epoch)
    writer.add_scalar('test/accuracy', acc, epoch)
    writer.add_scalar('test/error', 100-acc, epoch)
    if acc > best_acc:
        print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #         'optimizer': optimizer.state_dict(),
    #         'scheduler': scheduler.state_dict(),
    #     }
    #     # if not os.path.isdir('checkpoint'):
    #     #     os.mkdir('checkpoint')
    #     torch.save(state, os.path.join(logdir, 'best-{}-ckpt.t7'.format(model_name)))
        best_acc = acc


for epoch in range(start_epoch, int(30*3.5)):
    train(epoch)
    scheduler.step()
    val(epoch)
