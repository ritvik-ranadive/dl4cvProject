import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable

class FaceFront(nn.Module):
    """
    def __init__(self):
        super(FaceFront, self).__init__()
        # size = 1*64*64
        self.conv1 = nn.Sequential(nn.Dropout2d(0.2),
                                   nn.Conv2d(1, 32, 7),
                                   nn.ReLU())
        # size = 32*64*64
        self.max1 = nn.MaxPool2d(2, stride=2)
        # size = 32*32*32
        self.fc1 = nn.Sequential(nn.Dropout2d(0.5),
                                    nn.Linear(26912, 4096),
                                    nn.ReLU())

        self.conv2 = nn.Sequential(nn.Dropout2d(0.5),
                                   nn.Conv2d(1, 32, 5, stride=1, padding=2),
                                   nn.ReLU())
        # size = 32*32*32
        self.max2 = nn.MaxPool2d(3, stride=3)
        # size = 32*16*16

        # size = 32*16*16

        self.fc2 = nn.Linear(14112, 4096)
        # size = 1*64*64

    def forward(self, x):
        #print(x.size())

        x = self.conv1(x)


        x = self.max1(x)


        x = x.view(-1, self.num_flat_features(x))

        x = self.fc1(x)

        x = x.view(-1, 1, 64, 64)

        x = self.conv2(x)

        x = self.max2(x)

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc2(x)
        x = x.view(-1, 1, 64, 64)

        # print(x.size())
        return x
    """
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def __init__(self):
        super(FaceFront, self).__init__()
        # size = 1*64*64
        self.drop = nn.Dropout2d(0.2)

        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, stride=1, padding=2,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5))
        self.max1 = nn.MaxPool2d(2)#, return_indices=True)
        # size = 32*32*32

        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 5, stride=1, padding=2,bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(0.5))
        self.max2 = nn.MaxPool2d(2)# return_indices=True)
        # size = 32*16*16

        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, 5, stride=1, padding=2,bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Dropout2d(0.5))

        #self.max3 = nn.MaxPool2d(2)# return_indices=True)
        # size = 128*8*8

        #self.conv4 = nn.Sequential(nn.Conv2d(128, 128, 5, stride=1, padding=2,bias=True),
        #                           nn.ReLU(inplace=True),
         #                          nn.Dropout2d(0.5))

        #self.conv4 = nn.Conv2d(32, 1, 1, stride=1, padding=0, bias=True)
        #self.relu4 = nn.ReLU(inplace=True)
        #self.drop4 = nn.Dropout2d(0.5)
        # size = 32*16*16
        #self.convt1 = nn.ConvTranspose2d(32,16,4,stride=2,padding=1,bias=True)
        #self.convt2 = nn.ConvTranspose2d(16,1,4,stride=2,padding=1,bias=True)
        self.fc = nn.Linear(8192,4096,bias=True)
        # size = 1*64*64

    def forward(self, x):
        #print(x.size())
        x = self.drop(x)

        x = self.conv1(x)
        x = self.max1(x)


        x = self.conv2(x)
        x = self.max2(x)

        x = self.conv3(x)
        #x = self.max3(x)

       # x = self.conv4(x)

        x = x.view(-1, self.num_flat_features(x))
        #print(x.size())
        x=  self.fc(x)
        x = x.view(-1, 1,64,64)
        #x = self.maxun2(x,indexes2)
        #x = self.convt1(x)
        #x = self.maun1(x,indexes1)
        #x = self.convt2(x)
        #print(x.size())
        return x

