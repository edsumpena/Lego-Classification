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


class ImageClassifRNN(nn.Module):
    
    # input_size = image height
    # input_dim = RNN input layer dimension
    # rnn_hidden = rnn latent dimension size
    # classes = number of model output classes
    # stacked_rnn_layers = number of stacked RNN layers
    def __init__(self, input_size, input_dim, rnn_hidden, classes, stacked_rnn_layers: int = 1):
        super(ImageClassifRNN, self).__init__()

        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.activation = nn.ReLU()

        self.rnn = nn.RNN(input_dim, rnn_hidden, num_layers=stacked_rnn_layers)

        self.output_layer = nn.Linear(rnn_hidden, classes)

        self.input_size = input_size
        self.rnn_hidden = rnn_hidden
        self.stacked_rnn_layers = stacked_rnn_layers

    def forward(self, input: torch.Tensor):
        # batch_size x 3 x height x width
        x = self.activation(self.bn(self.conv(input)))

        # batch_size x height x width
        x = x.view(-1, self.input_size, self.input_size)
        batch_size = x.size(0)

        patch_size = 8

        x = x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).contiguous().view(batch_size, -1, patch_size**2)

        # num_rnn_steps x batch_size x image_size
        x = x.permute(1, 0, 2)

        # hidden = torch.zeros(self.stacked_rnn_layers, batch_size, self.rnn_hidden)

        # _ = the output features at each time step (don't need for classification)
        # hidden_output = final hidden state for each element in the batch
        # (stacked_rnn_layers x batch_size x rnn_hidden)
        _, hidden_output_rnn = self.rnn(x)

        # We only care about output from the final RNN layer
        out = self.output_layer(hidden_output_rnn[-1,:,:])

        return out # batch_size x classes
    

class ImageClassifLSTM(nn.Module):
    
    # input_size = image height
    # input_dim = RNN input layer dimension
    # rnn_hidden = rnn latent dimension size
    # classes = number of model output classes
    # stacked_rnn_layers = number of stacked RNN layers
    def __init__(self, input_size, input_dim, rnn_hidden, classes, stacked_rnn_layers: int = 1):
        super(ImageClassifLSTM, self).__init__()

        self.conv = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.activation = nn.ReLU()

        self.lstm = nn.LSTM(input_dim, rnn_hidden, num_layers=stacked_rnn_layers)

        self.output_layer = nn.Linear(rnn_hidden, classes)

        self.input_size = input_size
        self.rnn_hidden = rnn_hidden
        self.stacked_rnn_layers = stacked_rnn_layers

    def forward(self, input: torch.Tensor):
        # batch_size x 3 x height x width
        x = self.activation(self.bn(self.conv(input)))

        # batch_size x height x width
        x = x.view(-1, self.input_size, self.input_size)
        batch_size = x.size(0)

        patch_size = 8

        x = x.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).contiguous().view(batch_size, -1, patch_size**2)

        # num_rnn_steps x batch_size x image_size
        x = x.permute(1, 0, 2)

        # hidden = torch.zeros(self.stacked_rnn_layers, batch_size, self.rnn_hidden)

        # _ = the output features at each time step (don't need for classification)
        # hidden_output = final hidden state for each element in the batch
        # (stacked_rnn_layers x batch_size x rnn_hidden)
        _, hidden_output_rnn = self.lstm(x)

        # We only care about output from the final LSTM layer
        out = self.output_layer(hidden_output_rnn[0][-1,:,:])

        return out # batch_size x classes


def get_accuracy(logit, target, batch_size):
    ''' Obtain accuracy for training round '''
    corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
    accuracy = 100.0 * corrects/batch_size
    return accuracy.item()


def main():
    BATCH_SIZE = 64
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

    train_data = torchvision.datasets.ImageFolder(root='./Legos/train/', transform=trainTransform)
    train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_data = torchvision.datasets.ImageFolder(root='./Legos/val/', transform=trainTransform)
    val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    test_data = torchvision.datasets.ImageFolder(root='./Legos/test/', transform=testTransform)
    test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # RNN: 150 - 1
    # LSTM: 256 - 3
    RNN = ImageClassifRNN(32, 64, 150, 10, stacked_rnn_layers=1)
    # RNN = ImageClassifLSTM(32, 64, 256, 10, stacked_rnn_layers=5)
    optimizer = optim.Adam(RNN.parameters(), lr=0.001)

    RNN = RNN.to(device)

    EPOCHS = 200
    for epoch in range(1, EPOCHS + 1):
        train_running_loss = 0.0
        train_acc = 0.0

        for i, data in enumerate(train_data_loader):
            optimizer.zero_grad()

            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = RNN(images)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, BATCH_SIZE)

        eval_running_loss = 0.0
        eval_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0

        with torch.no_grad():
            for j, data in enumerate(val_data_loader):
                images, labels = data

                images = images.to(device)
                labels = labels.to(device)

                outputs = RNN(images)
                loss = F.cross_entropy(outputs, labels)

                eval_running_loss += loss.detach().item()
                eval_acc += get_accuracy(outputs, labels, BATCH_SIZE)

            for k, data in enumerate(test_data_loader):
                images, labels = data

                images = images.to(device)
                labels = labels.to(device)

                outputs = RNN(images)
                loss = F.cross_entropy(outputs, labels)

                test_loss += loss.detach().item()
                test_acc += get_accuracy(outputs, labels, BATCH_SIZE)

        print('[{}/{}]'.format(epoch, EPOCHS),
            'Training Loss: {:<8.3}'.format(train_running_loss / i),
            'Training Acc: {:<8.3}'.format(train_acc / i * 0.01),
            'Validation Loss: {:<8.3}'.format(eval_running_loss / j),
            'Validation Acc: {:<8.3}'.format(eval_acc / j * 0.01),
            'Test Loss: {:<8.3}'.format(test_loss / k),
            'Test Acc: {:<8.3}'.format(test_acc / k * 0.01))
            

if __name__=='__main__':
    main()