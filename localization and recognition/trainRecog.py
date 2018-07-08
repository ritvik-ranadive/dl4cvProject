from YouTubeFacesDB import YouTubeFacesDB
from YouTubeFacesDB.vgg import vgg16
import numpy as np
import torch.cuda
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pathlib2 import Path
import os

# Read the script directory, will come in handy later!
script_dir = os.path.dirname(__file__)

# Read the database file
# db = YouTubeFacesDB('/home/ritvik/dl4cv/ytfdb.h5')
db = YouTubeFacesDB(os.path.join(script_dir, "ytfdb.h5"))
# db = YouTubeFacesDB('/home/ritvik/testdata.h5')
db.split_dataset(validation_size=0.2, test_size=0.1)

# Open the file for recording logs
logFile = open("recognizerLogs.txt", "a")
trainLossFile = open("trainLoss.txt", "a")
valLossFile = open("valLoss.txt", "a")
trainAccFile = open("trainAcc.txt", "a")
valAccFile = open("valAcc.txt", "a")

# Check if a previously saved model exists
myNetwork = Path("recognizer.pt")
if myNetwork.is_file():
    print("Saved model exists...!!!")
    network = torch.load("recognizer.pt")
    if torch.cuda.is_available():
        network.cuda()
else:
    print("Creating model...!!!")
    # Create a file for storing names
    celebNames = db.labels
    fileName = "celebNames.txt"
    file = open(fileName, "w");
    p = 1
    for celebName in celebNames:
        writeText = str(p) + ";" + celebName + '\n'
        file.write(writeText)
        p = p + 1

    # Define the network and optimizer
    network = vgg16(True)
    for param in network.parameters():
        param.requires_grad = False
    # network.fc = nn.Linear(4096, 1000)
    network.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 1595),
    )
    if torch.cuda.is_available():
        network.cuda()
    print("Model created...!!!")

learning_rate = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.classifier.parameters(), learning_rate)

trainLoss = []
valLoss = []
trainAcc = []
valAcc = []

# Create a minibatch and train the network
epochs = 10
running_loss = 0.0
x = 1
numberOfSaves = 0
for epoch in range(epochs):
    for inputs_train, labels_train in db.generate_batches(batch_size=100, dset='train', rest=False):
        batch_size = 100
        #########################################
        ###########TRAIN THE MODEL###############
        #########################################
        # print(np.shape(inputs_train))
        # print(np.shape(labels_train))
        # Creating an array of class labels
        labelValues = [labels.tolist().index(1) for labels in labels_train]
        labelValues = np.array(labelValues)
        # print(labelValue)
        # print(np.shape(labels_train[:, x]))
        # print('Index: {}'.format(labels_train.index(1)))
        inputs_train = torch.from_numpy(inputs_train)
        labels_train = torch.from_numpy(labelValues)
        # print('Celeb Names: {}'.format(np.shape(celebNames)))
        if torch.cuda.is_available():
            inputs_train = Variable(inputs_train.cuda())
        else:
            inputs_train = Variable(inputs_train)
        optimizer.zero_grad()
        outputs_train = network(inputs_train)
        # print(outputs_train)
        # exit(0)
        # outputValues = [outputs.tolist().index(1) for outputs in outputs_train]
        labels_train = torch.LongTensor(labels_train)
        if torch.cuda.is_available():
            labels_train = Variable(labels_train.cuda())
        else:
            labels_train = Variable(labels_train)
        loss = criterion(outputs_train, labels_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        _, predictions_train = torch.max(outputs_train.data, 1)
        correct_train = (predictions_train == labels_train.data).sum()
        if (running_loss/x) < 0.01:
            break
        x = x + 1
        print('Training Loss: {}'.format(running_loss / (x + numberOfSaves)))
        logFile.write('Training Loss: {}'.format(running_loss / (x + numberOfSaves)))
        print('Training Accuracy: {}'.format(float(correct_train) / batch_size))
        logFile.write('Training Accuracy: {}'.format(float(correct_train) / batch_size))
        trainLoss.append(running_loss / (x + numberOfSaves))
        trainAcc.append(float(correct_train) / batch_size)
        # Save the network every 100 iterations
        if x % 50 == 0:
            #########################################
            ###########VALIDATE THE MODEL############
            #########################################
            for inputs_val, labels_val in db.generate_batches(batch_size=batch_size, dset='val', rest=False):
                inputs_val = torch.from_numpy(inputs_val)
                labelValues = [labels.tolist().index(1) for labels in labels_val]
                labelValues = np.array(labelValues)
                labels_val = torch.from_numpy(labelValues)
                if torch.cuda.is_available():
                    inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
                else:
                    inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
                optimizer.zero_grad()
                outputs_val = network(inputs_val)
                loss = criterion(outputs_val, labels_val)
                loss.backward()
                optimizer.step()
                running_loss += loss.data[0]
                _, predictions_val = torch.max(outputs_val.data, 1)
                # print('labelValues:{}'.format(labels_val.data))
                # print('predictions:{}'.format(predictions))
                correct_val = (predictions_val == labels_val.data).sum()
                print("-------------------------------------------------------------")
                print('Validation loss: {}'.format(running_loss / (x+numberOfSaves)))
                logFile.write('Validation loss: {}'.format(running_loss / (x+numberOfSaves)))
                print('Validation accuracy of system is: {}'.format(float(correct_val) / batch_size))
                logFile.write('Validation accuracy of system is: {}'.format(float(correct_val) / batch_size))
                valLoss.append(running_loss / (x+numberOfSaves))
                valAcc.append(float(correct_val) / batch_size)
                break
            #########################################
            #############STORE THE MODEL#############
            #########################################
            print('Saving network......')
            # if torch.cuda.is_available():
            #     network.cpu()
            torch.save(network, 'recognizer.pt')
            numberOfSaves += 1
            print('Network saved...!    {}'.format(numberOfSaves))
            for number in trainLoss:
                trainLossFile.write(str(number))
                trainLossFile.write("\n")
            for number in valLoss:
                valLossFile.write(str(number))
                valLossFile.write("\n")
            for number in trainAcc:
                trainAccFile.write(str(number))
                trainAccFile.write("\n")
            for number in valAcc:
                valAccFile.write(str(number))
                valAccFile.write("\n")
            trainLoss = []
            valLoss = []
            trainAcc = []
            valAcc = []
            print("------------------------------------------------------------")
    print('------------Epoch Complete-------------')
    logFile.write('------------Epoch Complete-------------')
# print('Epoch: {} of {} completed!'.format(epoch, epochs))