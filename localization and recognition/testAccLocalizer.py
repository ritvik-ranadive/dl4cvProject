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

# Read the script directory, will come in handy later!
script_dir = os.path.dirname(__file__)

# Check if a previously saved model exists
myNetwork = Path("localizer.pt")
if myNetwork.is_file():
    print("Saved model exists...!!!")
    network = torch.load("localizer.pt")
    if torch.cuda.is_available():
        network.cuda()

# Read the data
filePath = 'test.txt'
fileName = open(filePath, "r")
data = fileName.read()
data = data.split("\n")

# Data Structure to store accuracies and logs
testAcc = []
testLogFile = open("testLocalizerLogs.txt", "a")

for i in range(1, 700, 1):
    #########################################
    ###########TRAIN THE MODEL###############
    #########################################
    inputs, labels = generate_minibatch(data, 100)
    inputs_test = torch.from_numpy(inputs)
    labels_test = torch.from_numpy(labels)
    if torch.cuda.is_available():
        inputs_test, labels_test = Variable(inputs_test.cuda()), Variable(labels_test.cuda())
    else:
        inputs_test, labels_test = Variable(inputs_test), Variable(labels_test)
    outputs_test = network(inputs_test)
    if torch.cuda.is_available():
        labels_test = labels_test.data.cpu().numpy()
    else:
        labels_test = labels_test.data.numpy()
    labels_test = torch.FloatTensor(labels_test)
    # labels_train = torch.LongTensor(labels_train)
    if torch.cuda.is_available():
        labels_test = Variable(labels_test.cuda())
    else:
        labels_test = Variable(labels_test)
    # print('Outputs', outputs_train)
    # print('Labels', labels_train)
    testAccuracy = calculateAccuracy(outputs_test, labels_test, threshold=0.5)
    testAcc.append(testAccuracy)
    print('Training Accuracy: {}'.format(testAccuracy))
    testLogFile.write('Test Accuracy: {}'.format(testAccuracy))
    testLogFile.write("\n")
    # print('----------------------------------------------------')

for num in testAcc:
    testLogFile.write(str(num))
    testLogFile.write("\n")

# testAcc = open("testLocalizerLogs.txt","r").read()
# testAcc = testAcc.split("\n")
# testAcc = np.array(testAcc).astype(float)
# print(np.average(testAcc,0))
#Average Accuracy = 0.930450350006 IoU = 0.5