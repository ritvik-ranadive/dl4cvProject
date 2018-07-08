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

testAcc = []
testLogs = open("testRecognizerLogs.txt", "a")

# Check if a previously saved model exists
myNetwork = Path("recognizer.pt")
if myNetwork.is_file():
    print("Saved model exists...!!!")
    network = torch.load("recognizer.pt")
    if torch.cuda.is_available():
        network.cuda()
    #########################################
    ############TEST THE MODEL###############
    #########################################
    for inputs_test, labels_test in db.generate_batches(batch_size=100, dset='test', rest=False):
        inputs_test = torch.from_numpy(inputs_test)
        labelValues = [labels.tolist().index(1) for labels in labels_test]
        labelValues = np.array(labelValues)
        labels_test = torch.from_numpy(labelValues)
        if torch.cuda.is_available():
            inputs_test, labels_test = Variable(inputs_test.cuda()), Variable(labels_test.cuda())
        else:
            inputs_test, labels_test = Variable(inputs_test), Variable(labels_test)
        outputs_test = network(inputs_test)
        _, predictions_test = torch.max(outputs_test.data, 1)
        # print('labelValues:{}'.format(labels_val.data))
        # print('predictions:{}'.format(predictions))
        correct_test = (predictions_test == labels_test.data).sum()
        print('Test Accuracy: {}'.format(float(correct_test) / 100))
        testLogs.write('Test Accuracy: {}'.format(float(correct_test) / 100))
        testAcc.append(float(correct_test) / 100) 

# testAccFile = open("testRecognizerLogs.txt","r")
# testAcc = testAccFile.read()
# testAcc = testAcc.split("\n")
# testAcc = [accuracy[15:len(accuracy)] for accuracy in testAcc]
# testAcc = np.array(testAcc).astype(float)
# print np.average(testAcc,0)
# Final Accuracy = 0.969436392915