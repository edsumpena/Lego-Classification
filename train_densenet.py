#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

from torchvision import transforms
from torchvision.utils import save_image

import torchvision
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import os
import sys
import math

import shutil

import densenet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=64)
    parser.add_argument('--nEpochs', type=int, default=10)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', default='./ckpts4/')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normTransform
    ])
    testTransform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        # normTransform
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    print('Training set: ' + str(sum(1 for _, _, files in os.walk('./Legos/train/') for f in files)))
    print('Validation set: ' + str(sum(1 for _, _, files in os.walk('./Legos/val/') for f in files)))
    print('Test set: ' + str(sum(1 for _, _, files in os.walk('./Legos/test/') for f in files)))

    train_data = torchvision.datasets.ImageFolder(root='./Legos/train/', transform=trainTransform)
    train_data_loader = DataLoader(train_data, batch_size=args.batchSz, shuffle=True,  num_workers=4)
    val_data = torchvision.datasets.ImageFolder(root='./Legos/val/', transform=trainTransform)
    val_data_loader = DataLoader(val_data, batch_size=args.batchSz, shuffle=True,  num_workers=4)
    test_data = torchvision.datasets.ImageFolder(root='./Legos/test/', transform=testTransform)
    test_data_loader = DataLoader(test_data, batch_size=args.batchSz, shuffle=True,  num_workers=4)


    net = densenet.RDenseNetSE(growthRate=12, depth=100, reduction=0.5,
                            bottleneck=True, nClasses=10)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, train_data_loader, optimizer, trainF)
        test(args, epoch, net, val_data_loader, optimizer, testF, val=True)

        if epoch >= 10:
            test(args, epoch, net, test_data_loader, optimizer, testF, val=False)
        
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('python3 ./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        
        plt.figure(figsize=(20,9))
        plt.imshow(data.data.cpu().numpy()[0].T)
        plt.colorbar()
        plt.savefig('./out.png')
        plt.close()

        # print(target)

        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        # print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
        #     partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
        #     loss.data, err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data, err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF, val=True):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal

    if val:
        print('\Validation set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
            test_loss, incorrect, nTotal, err))

        testF.write('{},{},{}\n'.format(epoch, test_loss, err))
        testF.flush()
    else:
        print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
