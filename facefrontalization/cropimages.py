import zipfile
import numpy as np
import pandas as pd
import io
import yaml
import PyQt5
import math
import matplotlib
# Import Pillow:
from PIL import Image
matplotlib.use('qt5agg')
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def crop():
    print("start")
    filepath = "/home/armin/Downloads/put-face-database/"
    files = {"Images_031_040.zip", "Images_081_090.zip",
             "frontal_1_50.zip", "Images_041_050.zip", "Images_091_100.zip",
             "frontal_51_100.zip", "Images_051_060.zip",
             "Images_001_010.zip", "Images_061_070.zip",
             "Images_011_020.zip", "Images_071_080.zip"
             };
    # "contours.zip",  "Landmarks2.zip", "Subsets.zip", "regions2.zip"
    zf = zipfile.ZipFile(filepath + "regions2.zip")
    infos = {}
    for f in zf.namelist():

        if len(f) > 5:
            yml = zf.read(f)
            dataMap = (yaml.load(yml[11:]))
            # print(dataMap['Face'])
            infos[f[5:-4]] = dataMap['Face']
    print(len(infos))
    pics = 0
    for f in files:
        zf = zipfile.ZipFile(filepath + f)
        for picname in zf.namelist():
            if len(picname) > 5:
                picinfos = infos[picname[5:-4]]
                pic = zf.open(picname)
                img = Image.open(pic)
                # img = mpimg.imread(pic)
                # imgplot = plt.imshow(img)
                imgc = img.crop((picinfos['x'], picinfos['y'], picinfos['x'] + picinfos['width'],
                                 picinfos['y'] + picinfos['height']))
                # plt.show(imgplot)
                imgc.save("/tmp/images/" + picname[5:])
                pics = pics + 1

    print(pics, " ", len(infos))

def makegrey():
    filepath = "/home/armin/Downloads/put-face-database/"
    filename="croped-resized64x64_new.zip"
    zf = zipfile.ZipFile(filepath + filename)
    leng= 10#len(filename)+1
    for f in zf.namelist():
        if len(f) > leng:
            pic = zf.open(f)
            im = Image.open(pic)
            im = im.convert('L')
            im.save("/tmp/images/" + f[:-4]+'.bmp')



def makesquare():
    print("start")
    filepath = "/home/armin/normalised_faces/"
    filename = "cropped.zip"
    zf = zipfile.ZipFile(filepath + filename)
    leng = len(filename)-3
    for f in zf.namelist():
        if len(f) > leng:
            pic = zf.open(f)
            im = Image.open(pic)
            width, height = im.size  # Get dimensions
            size = min(width,height)
            left = math.ceil((width - size) / 2)
            top = math.ceil((height - size) / 2)
            right = math.floor((width + size) / 2)
            bottom = math.floor((height + size) / 2)
            if left-right < top-bottom:
                top -=1
            elif left-right > top-bottom:
                right-=1
            im = im.crop((left, top, right, bottom))
            size = 64,64
            im.thumbnail(size, Image.ANTIALIAS)
            print(im.size)
            #im = im.convert('L')
            im.save("/tmp/images/" + f[leng:-3]+"bmp")
def frontal():
    P = np.zeros((64, 64))
    for i in range(0, 32):
        P[i, i] = 1

    Q = np.zeros((64,64))
    for i in range(32, 64):
        Q[i, i] = 1
    gamma =1;
    filepath = "dataset/"
    filename = "croped-resized-grey.zip"
    zf = zipfile.ZipFile("/home/armin/Downloads/lfw/lfw-cropped-funnled-gray.zip")
    #zf = zipfile.ZipFile(filepath+filename)
    zippath ="lfw-cropped-funnled-gray/"
    lfwlen = len(zippath)

    #lfwlen = len("croped-resized-grey/faces/")
    values = {}
    for f in zf.namelist():
            if len(f) > lfwlen:
                if f[-1] == "/":
                    folder = f
                else:
                    v = []
                    if f[lfwlen:-9] in values:
                        v = values[f[lfwlen:-9]]
                    pic = zf.open(f)
                    Y = mpimg.imread(pic)
                    x = np.dot(Y,P)-np.dot(Y,Q)
                    x = np.linalg.norm(x,'fro')
                    v.append((f[lfwlen:],np.power(x,2) - gamma * np.linalg.norm(Y,'nuc')))
                    values[f[lfwlen:-9]] = v
                    print(f[lfwlen:],len(v))

    biggest = 0
    biggestvalue = ""
    for key,v in values.items():
        if len(v)>2:
            v.sort(key=lambda tup: tup[1])
            pic = zf.open(zippath+v[0][0])
            print(zippath+v[-1][0])
            Image.open(pic).save("/tmp/images/"+v[-1][0])

def crop_lfwdeepfunnled():

    zf = zipfile.ZipFile("/home/armin/Downloads/lfw/"+"lfw-deepfunneled.zip")
    folder = ""
    pics=0
    for picname in zf.namelist():
        if picname[-1] == "/":
            folder = picname
        else:
            pic = zf.open(picname)
            im = Image.open(pic)
            # img = mpimg.imread(pic)
            # imgplot = plt.imshow(img)
            im = im.crop((83, 92, 166,175))
            # plt.show(imgplot)

            #resize make squere
            width, height = im.size  # Get dimensions
            size = min(width, height)
            left = math.ceil((width - size) / 2)
            top = math.ceil((height - size) / 2)
            right = math.floor((width + size) / 2)
            bottom = math.floor((height + size) / 2)
            if left - right < top - bottom:
                top -= 1
            elif left - right > top - bottom:
                right -= 1
            im = im.crop((left, top, right, bottom))
            size = 64, 64
            im.thumbnail(size, Image.ANTIALIAS)
            #print(im.size)
            im = im.convert('L')

            im.save("/tmp/images/" + picname[len(folder):-3]+"bmp")
            pics = pics + 1

    print(pics)


def ytf_crop():
    path="/home/armin/Downloads/YouTubeFaces"
    zf = zipfile.ZipFile(path+"frame_images_DB.zip")
    folder = ""
    pics=0
    for picname in zf.namelist():
        if picname[-1] == "/":
            folder = picname
        else:
            print(picname)
            pic = zf.open(picname)
            im = Image.open(pic)

            #im = im.crop((left, top, right, bottom))
            size = 64, 64
            im.thumbnail(size, Image.ANTIALIAS)
            #print(im.size)
            im = im.convert('L')
            #print(picname.split("/"))
            im.save("/tmp/images/" + picname.split("/")[-1][:-3]+"bmp")
            pics = pics + 1

    print(pics)

def resizeedfunnled_frontal():

    zf = zipfile.ZipFile("/home/armin/Downloads/lfw/3dfunneld/LFW3D.0.1.1.zip")
    folder = ""
    pics=0
    for picname in zf.namelist():
        if picname[-1] == "/":
            folder = picname
        else:
            print(picname)
            pic = zf.open(picname)
            im = Image.open(pic)

            #im = im.crop((left, top, right, bottom))
            size = 64, 64
            im.thumbnail(size, Image.ANTIALIAS)
            #print(im.size)
            im = im.convert('L')
            #print(picname.split("/"))
            im.save("/tmp/images/" + picname.split("/")[-1][:-3]+"bmp")
            pics = pics + 1

    print(pics)

def cropimagestesttdcv():
    path ="/home/armin/Documents/Studium/Bachelor/9.Semester_17-18_WinterSemester/TDCV/tdcv/ex2/data/task3/"
    gt = "gt.zip"
    test = "test.zip"
    zf = zipfile.ZipFile(path+gt)
    folder = ""
    pics = 0
    postgt=len(".gt.txt")
    oregt= len("gt/")
    gts={}
    #load crop descriptor
    for picname in zf.namelist():
        if picname[-1] == "/":
            folder = picname
        else:
            print(picname)
            spamreader = pd.read_csv(zf.open(picname),delimiter=' ',header=None)
            for i in spamreader.values:
                key =picname[oregt:-postgt]
                print(key)
                if key not in gts:
                    gts[key] = []
                gts[key].append([[str(0) + str(i[0]), i[1:5]]])
    print(gts       )
    #load images
    zf = zipfile.ZipFile(path+test)
    for picname in zf.namelist():
        if picname[-1] == "/":
            folder = picname
        else:
            print(picname)
            pic = zf.open(picname)
            im = Image.open(pic)
            index = picname[len("test/"):-4]
            if index in gts:
                gtss = gts[index]
                print(gtss)
                for bla in gtss:
                    print(bla)
                    print( im.size )
                    imgc = im.crop((bla[0][1][0],bla[0][1][1],bla[0][1][2],bla[0][1][3]))
                    imgc.save("/tmp/images/" + bla[0][0]+"/n1"+index+".jpg")
                    pics = pics + 1


if __name__ == "__main__":
    makesquare()
