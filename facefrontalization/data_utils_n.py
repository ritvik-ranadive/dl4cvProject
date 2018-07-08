"""Data utility functions."""
import os

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import zipfile
import pickle as pickle
from PIL import Image

# pylint: disable=C0326
class OverfitSampler(object):
    """
    Sample dataset to overfit.
    """

    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples


class PutFaceDataset(data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, index):
        img = self.X[index]
        label = self.y[index]

        img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return len(self.y)
    """
frontalfaces= ["00012018", "00022015", "00032011", "00042016", "00052010", "00062011", "00072015", "00081013", "00091024",
               "00102016", "00112010", "00121022", "00131017", "00141016", "00152016", "00162010", "00171017",
               "00181016", "00191012", "00201017", "00212010", "00221016", "00231016", "00241016", "00251016",
               "00261013", "00272015", "00282010", "00292011", "00302011", "00311017", "00321016", "00322016",
               "00331017", "00341015", "00351016", "00365006", "00371017", "00381015", "00391016", "00401016",
               "00411013", "00421017", "00431015", "00441016", "00451014", "00461017", "00471016", "00481016",
               "00492011", "00501015", "00512011", "00522011", "00531016", "00541012", "00552010", "00561019",
               "00571016", "00582013", "00591017", "00601019", "00611015", "00622010", "00631019", "00642012",
               "00651019", "00661017", "00672019", "00681017", "00691014", "00701018", "00711014", "00721016",
               "00731016", "00742013", "00752010", "00761017", "00771016", "00782012", "00792011", "00801012",
               "00811016", "00822011", "00831020", "00842011", "00852013", "00861017", "00871015", "00881028",
               "00892015", "00901018", "00911016", "00921017", "00931016", "00942011", "00952009", "00961018",
               "00971016", "00982011", "00991016", "01001016"]
frontalgalsses=["00025008", "00065003", "00085008", "00155004", "00325015", "00495006", "00565010", "00815008", "00755006", "00545010"]
"""
frontalfaces = [12018, 22015, 32011, 42016,  52010,  62011,  72015,  81013,  91024,
                102016, 112010, 121022, 131017, 141016, 152016, 162010, 171017, 181016, 191012,
                201017, 212010, 221016, 231016, 241016, 251016, 261013, 272015, 282010, 292011,
                302011, 311017, 321016, 331017, 341015, 351016, 365006, 371017, 381015, 391016,
                401016, 411013, 421017, 431015, 441016, 451014, 461017, 471016, 481016, 492011,
                501015, 512011, 522011, 531016, 541012, 552010, 561019, 571016, 582013, 591017,
                601019, 611015, 622010, 631019, 642012, 651019, 661017, 672019, 681017, 691014,
                701018, 711014, 721016, 731016, 742013, 752010, 761017, 771016, 782012, 792011,
                801012, 811016, 822011, 831020, 842011, 852013, 861017, 871015, 881028, 892015,
                901018, 911016, 921017, 931016, 942011, 952009, 961018, 971016, 982011, 991016,
                1001016]
frontalgalsses = [25008, 65003, 85008, 155004, 325015, 495006, 565010, 815008, 855006, 755006, 545010]
frontalgalssesindex=[2, 6, 8, 15, 32, 49, 56, 81, 85, 75, 54]
def lfw_loadimages():
    filepath = "dataset/"
    frontal = "lfw-croped-frontal-gray.zip"
    trainingzip = "lfw-funnled-cropped-gray.zip"
    zf = zipfile.ZipFile(filepath + frontal)
    zipfilenamelen=len(frontal)
    to_tensor = transforms.ToTensor()
    numberfaces=0
    numberpersons=0
    persontargets={}
    for f in zf.namelist():
                #print(f[:-9])
                pic = zf.open(f)
                img = Image.open(pic)
                img.load()
                img = to_tensor(img)
                numberpersons += 1
                persontargets[f] = img
    print(len(persontargets))
    zff = zipfile.ZipFile(filepath + trainingzip)
    faces = torch.Tensor()
    targets = torch.Tensor()
    for f in zff.namelist():
        #print(f[:-9])
        pic = zff.open(f)
        img = Image.open(pic)
        img.load()
        img = to_tensor(img)
        numberfaces += 1
        if f in persontargets:
            faces = torch.cat((faces,img.view(1,1,64,64)),0)
            targets = torch.cat((targets,persontargets[f].view(1,1,64,64)),0)
    print(faces.size(),targets.size())
    return (faces,targets)

def colorferet_loadimages():
    filepath = "dataset/"
    frontal = "target-colfert.zip"
    trainingzip = "input-colfert.zip"
    zf = zipfile.ZipFile(filepath + frontal)
    zipfilenamelen=len(frontal)
    to_tensor = transforms.ToTensor()
    numberfaces=0
    numberpersons=0
    persontargets={}
    print("readtarget")
    for f in zf.namelist():
                #print(f[:-9])
                pic = zf.open(f)
                img = Image.open(pic)
                img.load()
                img = to_tensor(img)
                numberpersons += 1
                if '_fa' in f:
                    f = f.split('_')
                    f = f[0]+'_'+f[1]
                    persontargets[f] = img
    print(len(persontargets))
    zff = zipfile.ZipFile(filepath + trainingzip)
    faces = torch.Tensor()
    targets = torch.Tensor()
    print("readinput")
    for f in zff.namelist():
        #print(f[:-9])
        pic = zff.open(f)
        img = Image.open(pic)
        img.load()
        img = to_tensor(img)
        numberfaces += 1
        f = f.split('_')
        f = f[0][1:]+'_' + f[1]
        if f in persontargets:
            faces = torch.cat((faces,img.view(1,1,64,64)),0)
            targets = torch.cat((targets,persontargets[f].view(1,1,64,64)),0)
        else:
            print(f)
    print(faces.size(),targets.size())
    return (faces,targets)

def loadimages(number =-1):
    filepath = "dataset/"
    #filepath = "/home/armin/Downloads/put-face-database/"
    filename = "croped-resized-grey.zip"
    #filename = "croped-resized.zip"
    zf = zipfile.ZipFile(filepath + filename)
    leng = len("croped-resized-grey/faces/")
    glasses = [[]] * len(frontalgalsses)
    glassestargets = [None] * len(frontalgalsses)

    faces = [[]] * len(frontalfaces)
    facetargets = [None] * len(frontalfaces)
    numberglasses = 0
    numberfaces = 0
    numberpersons_faces=0
    numberpersons_glasses=0
    to_tensor = transforms.ToTensor()
    single = False
    if number != -1:
        single = True

    for f in zf.namelist():
        if len(f) > leng + 5:
            pic = zf.open(f)
            img = Image.open(pic)
            img.load()
            img = to_tensor(img)
            if "faces" in f:
                numberfaces += 1
                f = f[leng:-4]
                #print("face", f)
                index = int(f[:4]) - 1
                if number == index:
                    if int(f) not in frontalfaces:
                        if faces[index] == []:
                            faces[index] = [img]
                        else:
                            a = faces[index]
                            a.append(img)
                            faces[index] = a
                    else:
                        print("notin",index,f)
                        numberpersons_faces += 1
                        facetargets[index] = img
            elif "glasses" in f:
                numberglasses += 1
                f = f[leng + 2:-4]
                index = frontalgalssesindex.index(int(f[:4]))
                #print("glasses", f, index)
                if int(f) not in  frontalgalsses:
                    if glasses[index] == []:
                        glasses[index] = [img]
                        #print(img.size())
                    else:
                        a = glasses[index]
                        a.append( img)
                        glasses[index] = a
                else:
                    numberpersons_glasses += 1
                    print("notin", index, f)
                    glassestargets[index] = img
    print("persons: ",numberpersons_faces,"faces: ",numberfaces,"glasses: ",numberpersons_glasses,numberglasses)
    for i,f in enumerate(facetargets):
        if f is None:
            print ("None: ",i)
    return ((faces, facetargets), (glasses, glassestargets))

def imagesToDataSetTupels(data,amountoffaces = 100, numberOfPicture=100):
    inputs = []
    faces,facetargets = data
    facescounter =0
    for i,flist in enumerate(faces):
        if facescounter < amountoffaces:
            facescounter +=1
            facesNrcounter = 0
            for f in flist:
                if facesNrcounter < numberOfPicture:
                    facesNrcounter += 1
                    inputs.append((f,facetargets[i]))

    return inputs

def imagesToDataSetArray(data,amountoffaces = 100, numberOfPicture=100):
    inputs = []
    target = []
    faces,facetargets = data
    facescounter =0
    for i,flist in enumerate(faces):
        if facescounter < amountoffaces:
            facescounter += 1
            facesNrcounter = 0
            for f in flist:
                if facesNrcounter < numberOfPicture:
                    facesNrcounter += 1
                    inputs.append(f)
                    target.append(facetargets[i])

    return (inputs,target)

def imagesToDataSetTensor(data,amountoffaces = 100, numberOfPicture=100):
    inputs = torch.Tensor()
    targets = torch.Tensor()
    faces,facetargets = data
    facescounter =0

    for i,flist in enumerate(faces):
        if facescounter < amountoffaces:
            facescounter +=1
            facesNrcounter = 0
            #print(i)
            for f in flist[1:]:
                if facesNrcounter < numberOfPicture:
                    facesNrcounter += 1
                    inputs = torch.cat((inputs, f.view(1, 1, 64, 64)), 0)
                    targets = torch.cat((targets, facetargets[i].view(1, 1, 64, 64)), 0)
                    #inputs.append(f)
                    #target.append()
    print(inputs.size(),targets.size())
    return (inputs,targets)
def getsample():
    print("is doing")
def imgtopickl():
    with open('filename.pickle', 'wb') as handle:
        pickle.dump((loadimages()), handle, protocol=pickle.HIGHEST_PROTOCOL)


def getpickle():
    """

    :return:
    2 tupel faces and glasses
    """
    with open('filename.pickle', 'rb') as handle:
        b = pickle.load(handle)
    faces = b[0]
    glasses = b[1]
    return faces,glasses


def scoring_function(x, lin_exp_boundary, doubling_rate):
    assert np.all([x >= 0, x <= 1])
    score = np.zeros(x.shape)
    lin_exp_boundary = lin_exp_boundary
    linear_region = np.logical_and(x > 0.1, x <= lin_exp_boundary)
    exp_region = np.logical_and(x > lin_exp_boundary, x <= 1)
    score[linear_region] = 100.0 * x[linear_region]
    c = doubling_rate
    a = 100.0 * lin_exp_boundary / np.exp(lin_exp_boundary * np.log(2) / c)
    b = np.log(2.0) / c
    score[exp_region] = a * np.exp(b * x[exp_region])
    return score

def rel_error(x, y):
    """ Returns relative error """
    assert x.shape == y.shape, "tensors do not have the same shape. %s != %s" % (x.shape, y.shape)
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


#imgtopickl()
#((faces, facetargets), (glasses, glassestargets)) = loadimages()
#(f_train,f_val) = imagesToDataSet((faces,facetargets))