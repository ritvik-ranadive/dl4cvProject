'''
This file captures does the following:
    Takes the name of a celebrity as the input
    Choose the frames of the celebrity
    Pushes the celebrity's pictures through th following pipe:

            ----->localizer_cpu.pt ---- get bounding box on face ---
Frame -----|                                                       |-----> display frame with bounding box and name
            ----->recognizer_cpu.pt --- get celebrity name --------
'''

# import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.image as mpimg
from torch.autograd import Variable
import matplotlib.patches as patches

recognizer = torch.load("/home/ritvik/dl4cv/models/recognizer_cpu.pt")
localizer = torch.load("/home/ritvik/dl4cv/models/localizer_cpu.pt")

# Load the celeb names for each label
celebList = open("/home/ritvik/dl4cv/models/celebNames.txt", "r").read()
celebList = celebList.split("\n")
celebList = [celeb.split(";") for celeb in celebList]

celebName = "Tom Cruise"
celebName = celebName.replace(" ","_")
framesPath = "/home/ritvik/dl4cv/YouTubeFaces/frame_images_DB/"
labeledFacesPath = "/home/ritvik/dl4cv/YouTubeFaces/frame_images_DB/" + celebName + ".labeled_faces.txt"

frames = open(labeledFacesPath,"r").read()
frames = frames.split("\n")
frames = [frame.split(",") for frame in frames]
# print frames

i = 0
for frame in frames[1:20]:
    i += 1
    # Sending the frame to recognizer
    image = Image.open(framesPath + frame[0].replace("\\","/"))
    x1 = int(int(frame[2]) - (int(frame[4])/2))
    y1 = int(int(frame[3]) - (int(frame[5])/2))
    x2 = int(x1 + int(frame[4]))
    y2 = int(y1 + int(frame[5]))
    imageNetwork = image.crop((x1,y1,x2,y2))
    imageNetwork = imageNetwork.resize((224,224))
    # plt.imshow(imageNetwork)
    # plt.pause(0.5)
    imageNetwork = np.array(imageNetwork).astype('float32') / 255
    imageNetwork = imageNetwork.swapaxes(0,2)
    # print(np.shape(imageNetwork))
    imageNetwork = [imageNetwork]
    imageNetwork = np.array(imageNetwork)
    imageNetwork = torch.from_numpy(imageNetwork)
    imageNetwork = Variable(imageNetwork)
    output = recognizer(imageNetwork)
    _, prediction = torch.max(output.data, 1)
    nameToPrint = celebList[np.array(prediction)[0]][1]
    # print(celebList[np.array(prediction)[0]])

    # Sending frame to localizer
    imgNetwork = Image.open(framesPath + frame[0].replace("\\","/"))
    xSize = np.shape(imgNetwork)[1]
    ySize = np.shape(imgNetwork)[0]
    x = int(frame[2]) - (int(frame[4]) / 2)
    y = int(frame[3]) - (int(frame[5]) / 2)
    width = int(frame[4])
    height = int(frame[5])
    # Creating x,y,width,height at 224 X 224
    width224 = int(width * 224 / xSize)
    height224 = int(height * 224 / ySize)
    x224 = int(x * 224 / xSize)
    y224 = int(y * 224 / ySize)
    labels = [x224, y224, width224, height224]
    imgNetwork = imgNetwork.resize((224, 224))
    img_data = np.array(imgNetwork).astype('float32') / 255
    img_data = img_data.swapaxes(0, 2)
    img_data = [img_data]
    img_data = np.array(img_data)
    img_data = torch.from_numpy(img_data)
    img_data = Variable(img_data)
    output = localizer(img_data)
    output = output.data.numpy()
    output = output[0]
    xNet = int((output[0] * xSize) / 224)
    yNet = int((output[1] * ySize) / 224)
    widthNet = int((output[2] * xSize) / 224)
    heightNet = int((output[3] * ySize) / 224)

    # Save the image
    image.save(str(i)+"og.jpg", "JPEG")
    draw = ImageDraw.Draw(image)
    draw.rectangle(((xNet, yNet), (xNet+widthNet, yNet+heightNet)))
    draw.text((xNet+widthNet, yNet+heightNet), nameToPrint)
    image.save(str(i)+".jpg", "JPEG")
