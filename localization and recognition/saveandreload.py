from YouTubeFacesDB import YouTubeFacesDB
from YouTubeFacesDB.vgg import vgg16
import numpy as np
import torch.cuda
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from pathlib2 import Path


# Check if a previously saved model exists
myNetwork1 = Path("localizer.pt")
myNetwork2 = Path("recognizer.pt")
if myNetwork1.is_file():
    print("Saved model exists...!!!")
    network = torch.load("localizer.pt")
    network.cuda()
    print('Model loaded...!')
    print('Saving network......')
    network.cpu()
    torch.save(network, 'localizer_cpu.pt')
    print('Network saved...!')
    print("------------------------------------------------------------")
if myNetwork2.is_file():
    print("Saved model exists...!!!")
    network = torch.load("recognizer.pt")
    network.cuda()
    print('Model loaded...!')
    print('Saving network......')
    network.cpu()
    torch.save(network, 'recognizer_cpu.pt')
    print('Network saved...!')
    print("------------------------------------------------------------")
