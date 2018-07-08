from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable
import torch.cuda
import torch.nn.functional as F
import torch.nn as nn
import torch.optim
import random
class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={}):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.accurancy_test=[]
        self.loss=[]
        self.loss_test=[]
        #self.loss_func = lossDefinition
        #self.loss_func = torch.nn.CrossEntropyLoss2d(ignore_index=-1)
        #self.loss_func = torch.nn.NLLLoss2d()
        #self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []
    def printloss(self):
        print(self.loss_test)
    def train(self, model, data, testdata, num_epochs=100, epochsize=100):

        #loss = self.loss_func
        #optim = self.optim(model.parameters(), **self.optim_args)
        optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay=0.9)
        optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.9)
        #criterion = nn.CrossEntropyLoss()
        #criterion = nn.MSELoss(size_average=False)
        #criterion = nn.MultiLabelSoftMarginLoss()
        criterion = nn.SmoothL1Loss(size_average=False)
        #criterion = nn.BCELoss(size_average=False)
        self._reset_histories()
        iter_per_epoch = len(data)
        inputs, labels = (data[0]), (data[1])
        test, testlabels = (testdata[0]), (testdata[1])
        print("inputs: ",inputs.size(),"targets: ",labels.size())

        print('START TRAIN.')
        iterations = int(inputs.size(0)/epochsize)
        for epoch in range(num_epochs):
            running_loss = 0.0
            for iter in range(iterations):
                randoms = random.sample(range(0,inputs.size(0)),epochsize)
                input=torch.Tensor()
                label=torch.Tensor()
                for i in randoms:
                    input = torch.cat((input,inputs[i].view(1,1,64,64)),0)
                    label = torch.cat((label, (labels[i].view(1,1,64,64))),0)

                if torch.cuda.is_available():
                    model = model.cuda()
                    input, label = (input.cuda()), (label.cuda())
                input = Variable(input)
                label = Variable(label)

                optim.zero_grad()
                output = model(input)

                loss = criterion(output, label)
                #own loss function
                #frobinput = torch.add(label,bothbothboth torch.mul(output,-1)) #wrong function
                #loss = torch.sum(torch.sum((torch.abs(frobinput))))
                #frob = torch.rsqrt(torch.sum(torch.sum(torch.pow(torch.abs(frobinput),2))))
                #loss = torch.sum(torch.sum(frob*frob))
                #end own loss function
                loss.backward()
                running_loss += loss.data[0]
                optim.step()
                print("epoche: ", epoch,"iter:",iter, "loss:", loss.data[0])

            torch.save(model.cpu(), 'facefronter.pt')
            #accurency
            epochtraining=0
            predicted = 1
            predicted_right = 0;
            av_loss=0

            randoms = random.sample(range(0, test.size(0)), epochsize*2)
            input = torch.Tensor()
            label = torch.Tensor()
            for i in randoms:
                input =test[i].view(1, 1, 64, 64)
                label = (testlabels[i].view(1, 1, 64, 64))

                if torch.cuda.is_available():
                    model = model.cuda()
                    input, label = (input.cuda()), (label.cuda())
                input = Variable(input)
                label = Variable(label)

                optim.zero_grad()
                output = model(input)

                loss = criterion(output, label)
                predicted+=1;
                av_loss += loss.data[0]
                if loss.data[0] < 80 :
                    predicted_right +=1

            self.accurancy_test.append(predicted_right/predicted)
            self.loss_test.append(av_loss/predicted)
            self.loss.append(running_loss/iterations)

            print("epoche: ", epoch, "loss:", running_loss/iterations,"accurancy:",predicted_right/predicted,av_loss/predicted)

        print('Finished Training')
