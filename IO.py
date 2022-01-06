import cv2
import numpy as np
import glob

def ReadDataSet():
    x = []
    y = []
    #read database   
    for i in range(1,10):
        for filename in sorted(glob.glob('./database/' + str(i) + '/*.jpg')):
            img = cv2.imread(filename) 
            x.append(img)
            y.append(i)

    x= np.array(x)
    y= np.array(y)
    return x,y