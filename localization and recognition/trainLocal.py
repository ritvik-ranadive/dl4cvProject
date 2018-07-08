from PIL import Image
import os
import random as ran
from YouTubeFacesDB.vgg import vgg16
import numpy as np
import torch.cuda
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pathlib2 import Path


# Read the script directory, will come in handy later!
script_dir = os.path.dirname(__file__)

# Open the file for recording logs
logFile = open("localizerLogs.txt", "a")
trainLossFile = open("trainLoss.txt", "a")
valLossFile = open("valLoss.txt", "a")
trainAccFile = open("trainAcc.txt", "a")
valAccFile = open("valAcc.txt", "a")


# Check if a previously saved model exists
myNetwork = Path("localizer.pt")
if myNetwork.is_file():
    print("Saved model exists...!!!")
    network = torch.load("localizer.pt")
    if torch.cuda.is_available():
        network.cuda()
else:
    # Define the network and optimizer
    print("Creatig model....!")
    network = vgg16(True)
    for param in network.parameters():
        param.requires_grad = False
    network.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4),
        # nn.ReLU(True),
        # nn.Dropout(),
        # nn.Linear(4096, 1595),
        # nn.ReLU(True),
        # nn.Dropout(),
        # nn.Linear(1595, 4),
    )
    if torch.cuda.is_available():
        network.cuda()
    print("Model Created...!!!")

learning_rate = 1e-4
criterion = nn.SmoothL1Loss()
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.classifier.parameters(), learning_rate)

# Code to generate a minibatch
def generate_minibatch(data, batch_size=100):
    filePath = '/home/ritvik/YouTubeFaces/frame_images_DB/'
    resizedData = []

    #Choosing a random sample from all of train data
    indices = list(range(np.shape(data)[0]-1))
    ran.shuffle(indices)
    sample = [data[x].split(',') for x in indices[0:batch_size]]
    sample = [sam for sam in sample if len(sam[0]) != 0]
    images = []
    # labels = np.zeros((np.shape(sample)[0], 256, 256))

    for element in sample:
        imgNetwork = Image.open(filePath + element[0].replace("\\", "/"))
        xSize = np.shape(imgNetwork)[1]
        ySize = np.shape(imgNetwork)[0]
        element[4] = int(int(element[4]) * 224 / xSize)
        element[5] = int(int(element[5]) * 224 / ySize)
        element[2] = int((int(element[2]) * 224 / xSize) - (element[4] / 2))
        element[3] = int((int(element[3]) * 224 / ySize) - (element[5] / 2))
        resizedData.append([element[2], element[3], element[4], element[5]])
        # labels[element[2]:element[2]+element[4], element[3]:element[3]+element[5]] = 1
        imgNetwork = imgNetwork.resize((224, 224))
        img_data = np.array(imgNetwork).astype('float32') / 255
        img_data = img_data.swapaxes(0, 2)
        img_data = np.array(img_data)
        images.append(img_data)
    images = np.array(images)
    resizedData = np.array(resizedData)
    return images, resizedData

def calculateAccuracy(predicted, actual, threshold):
    accuracy = 0.0
    accurateRectangles = 0
    # print('Predicted Size: {}'.format(predicted.size()))
    # print('Actual Size: {}'.format(actual.size()))
    predicted = np.array(predicted.cpu().data)
    actual = np.array(actual.cpu().data)

    predictedRectangles = [[output[0], output[1], output[0]+output[2], output[1]+output[3]]
                           for output in predicted]
    actualRectangles = [[label[0], label[1], label[0] + label[2], label[1] + label[3]]
                           for label in actual]
    # print('predictedRectangles',predictedRectangles)
    # print('actualRectangles',actualRectangles)


    for i in range(np.shape(predictedRectangles)[0]):
        x1max = int(np.max([predictedRectangles[i][0], actualRectangles[i][0]]))
        y1max = int(np.max([predictedRectangles[i][1], actualRectangles[i][1]]))
        x2min = int(np.min([predictedRectangles[i][2], actualRectangles[i][2]]))
        y2min = int(np.min([predictedRectangles[i][3], actualRectangles[i][3]]))
        # print(x1max, y1max, x2min, y2min)
        if x1max >= x2min or y1max >= y2min:
            accurateRectangles += 0
        else:
            intersectArea = (x2min - x1max +1) * (y2min - y1max + 1)
            # print('intArea',intersectArea)
            predictedArea = (predictedRectangles[i][2] - predictedRectangles[i][0] + 1) * (predictedRectangles[i][3] - predictedRectangles[i][1] + 1)
            # print('preArea',predictedArea)
            actualArea = (actualRectangles[i][2] - actualRectangles[i][0] + 1) * (actualRectangles[i][3] - actualRectangles[i][1] + 1)
            # print('actArea', actualArea)
            iou = intersectArea/(predictedArea+actualArea-intersectArea)
            if iou >= threshold:
                accurateRectangles += 1
            else:
                accurateRectangles += 0
                # print('accurateRects:',accurateRectangles)

    accuracy = float(accurateRectangles)/np.shape(predictedRectangles)[0]
    return accuracy

# Read the data
filePath = 'train.txt'
fileName = open(filePath, "r")
data = fileName.read()
data = data.split("\n")
# data = data[0:len(data)-2]

# print(np.shape(inputs))
# print(np.shape(labels))

trainLoss = []
valLoss = []
trainAcc = []
valAcc = []

running_loss = 0.0
epochs = 1
numberOfSaves = 0
# for x in range(1, epochs, 1):
for i in range(1, 7000, 1):
    #########################################
    ###########TRAIN THE MODEL###############
    #########################################
    inputs, labels = generate_minibatch(data, 100)
    inputs_train = torch.from_numpy(inputs)
    labels_train = torch.from_numpy(labels)
    if torch.cuda.is_available():
        inputs_train, labels_train = Variable(inputs_train.cuda()), Variable(labels_train.cuda())
    else:
        inputs_train, labels_train = Variable(inputs_train), Variable(labels_train)
    optimizer.zero_grad()
    outputs_train = network(inputs_train)
    if torch.cuda.is_available():
        labels_train = labels_train.data.cpu().numpy()
    else:
        labels_train = labels_train.data.numpy()
    labels_train = torch.FloatTensor(labels_train)
    # labels_train = torch.LongTensor(labels_train)
    if torch.cuda.is_available():
        labels_train = Variable(labels_train.cuda())
    else:
        labels_train = Variable(labels_train)
    # print('Outputs', outputs_train)
    # print('Labels', labels_train)
    loss = criterion(outputs_train, labels_train)
    loss.backward()
    optimizer.step()
    running_loss += loss.data[0]
    print('Training Loss: {}'.format(running_loss / (numberOfSaves + i)))
    logFile.write('Training Loss: {}'.format(running_loss / (numberOfSaves + i)))
    trainLoss.append(running_loss / (numberOfSaves + i))
    trainAccuracy = calculateAccuracy(outputs_train, labels_train, threshold=0.5)
    trainAcc.append(trainAccuracy)
    print('Training Accuracy: {}'.format(trainAccuracy))
    logFile.write('Training Accuracy: {}'.format(trainAccuracy))
    # print('----------------------------------------------------')
    if i % 100 == 0:
        #########################################
        ###########VALIDATE THE MODEL############
        #########################################
        numberOfSaves += 1
        print('---------VALIDATION----------------')
        valPath = 'val.txt'
        valName = open(valPath, "r")
        valData = valName.read()
        valData = valData.split("\n")
        inputs_val, labels_val = generate_minibatch(valData, 100)
        # print(np.shape(inputs_val))
        # print(np.shape(inputs_val))
        inputs_val = torch.from_numpy(inputs_val)
        labels_val = torch.from_numpy(labels_val)
        if torch.cuda.is_available():
            inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
        else:
            inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)
        optimizer.zero_grad()
        outputs_val = network(inputs_val)
        if torch.cuda.is_available():
            labels_val = labels_val.data.cpu().numpy()
        else:
            labels_val = labels_val.data.numpy()
        labels_val = torch.FloatTensor(labels_val)
        # labels_train = torch.LongTensor(labels_train)
        if torch.cuda.is_available():
            labels_val = Variable(labels_val.cuda())
        else:
            labels_val = Variable(labels_val)
        _, predictions = torch.max(outputs_val.data, 1)
        loss = criterion(outputs_val, labels_val)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        # correct = (predictions == labels_val.data).sum()
        # print("-------------------------------------------------------------")
        print('Validation Loss: {}'.format(running_loss / (numberOfSaves + i)))
        logFile.write('Validation Loss: {}'.format(running_loss / (numberOfSaves + i)))
        valLoss.append(running_loss / (numberOfSaves + i))
        valAccuracy = calculateAccuracy(outputs_train, labels_train, threshold=0.5)
        valAcc.append(valAccuracy)
        print('Validation Accuracy: {}'.format(valAccuracy))
        logFile.write('Validation Accuracy: {}'.format(valAccuracy))
        valAcc.append(valAccuracy)
        #########################################
        #############STORE THE MODEL#############
        #########################################
        print('---------SAVE-----------')
        # if torch.cuda.is_available():
        #     network.cpu()
        torch.save(network, 'localizer.pt')
        print('Network saved...! {}'.format(numberOfSaves))
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
