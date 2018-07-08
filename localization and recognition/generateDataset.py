import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as pat
import numpy as np
from PIL import Image
import glob
import os
import random as ran

# plt.interactive(True)
# fig, ax = plt.subplots(1)
# filePath = '/home/ritvik/Pictures/2.535.jpg'
# image = Image.open(filePath)
# shape = np.shape(image)
# print(shape)
# image = image.resize((224, 224))
# newX = 97*224/shape[1]
# newY = 50*224/shape[0]
# newSizeX = 90*224/shape[1]
# newSizeY = 90*224/shape[0]
# ax.imshow(image)
# rect = plt.Rectangle((newX, newY), newSizeX, newSizeY, linewidth=1, angle=0, fill=False)
# # rect = plt.Rectangle((97, 50), 90, 90, linewidth=1, angle=0, fill=False)
# ax.add_patch(rect)
# plt.show()
# plt.pause(10)


def split_data(frameData, valSize=0.2, testSize=0.1):
    totalSize = np.shape(frameData)[0]
    # print(totalSize)
    # print(range(totalSize))
    valSize = int(valSize * totalSize)
    testSize = int(testSize * totalSize)
    # trainSize = totalSize - [valSize + testSize]
    indices = list(range(totalSize))
    # print(indices)
    ran.shuffle(indices)
    # print(indices)
    testData = [frameData[x] for x in indices[0:testSize]]
    valData = [frameData[x] for x in indices[testSize:testSize+valSize]]
    trainData = [frameData[x] for x in indices[testSize+valSize:totalSize]]
    return trainData, valData, testData

# Read the data
filePath = '/home/ritvik/dl4cv/YouTubeFaces/frame_images_DB/'
filesList = [fileName for fileName in os.listdir(filePath) if fileName.__contains__('labeled_faces.txt')]
# print(np.shape(filesList))
# print(filesList)
# exit(0)
frameData = []
for fileName in filesList:
    fileName = open(filePath + fileName, "r")
    text = fileName.read()
    text = text.split('\n')
    frameData += text
    fileName.close()
# print(np.shape(frameData))
frameData = [string.split(',') for string in frameData]
trainData, valData, testData = split_data(frameData, valSize=0.2, testSize=0.1)
print(np.shape(trainData))
print(np.shape(valData))
print(np.shape(testData))
# print(np.shape(frameData))
fileTrain = open('/home/ritvik/dl4cv/youtubefaces2/train.txt', "w")
for line in trainData:
    line = str(line).replace("'", "")
    fileTrain.write(line[1:len(line)-1])
    fileTrain.write("\n")
fileTrain.close()
fileVal = open('/home/ritvik/dl4cv/youtubefaces2/val.txt', "w")
for line in valData:
    line = str(line).replace("'", "")
    fileVal.write(line[1:len(line)-1])
    fileVal.write("\n")
fileVal.close()
fileTest = open('/home/ritvik/dl4cv/youtubefaces2/test.txt', "w")
for line in testData:
    line = str(line).replace("'", "")
    fileTest.write(line[1:len(line)-1])
    fileTest.write("\n")
fileTest.close()

