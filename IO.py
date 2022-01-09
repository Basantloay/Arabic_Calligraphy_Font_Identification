import cv2
import numpy as np
import glob
import os
import skimage.io as io
from skimage.util import dtype
from Preproccessing import *
import matplotlib.pyplot as plt

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def ReadDataSet():
    x = []
    y = []
    #read database   
    for i in range(1,10):
        for filename in sorted(glob.glob('./database/' + str(i) + '/*.jpg')):
            img = cv2.imread(filename) 
            x.append(img)
            y.append(i)

    x= np.array(x,dtype=object)
    y= np.array(y)
    return x,y

def ReadTestSet(directory):
    x_test = []
    # for filename in sorted(glob.glob(directory +'*.png')):
    #     img = cv2.imread(filename) 
    #     x_test.append(img)
    path = "./test"
    i=0
    list_dir=[int(file.split(".")[0]) for file in os.listdir(path)]
    list_dir.sort()
    for fname in list_dir:    
        img = cv2.imread(path + '/' + str(fname)+".png")
        x_test.append(img)
    i += 1


    return x_test

def WriteOutputFiles(directory,results,runningTime):
    resultFile = open(directory+'results.txt', 'w+')
    timeFile = open(directory+'times.txt', 'w+')
    # write to results file
    for i in range(len(results)-1):
        resultFile.write(str(int(results[i]))+'\n')  # python will convert \n to os.linesep
        timeFile.write(str(runningTime[i])+'\n')
    
    resultFile.write(str(int(results[-1])))
    timeFile.write(str(runningTime[-1]))
    resultFile.close()
    timeFile.close
