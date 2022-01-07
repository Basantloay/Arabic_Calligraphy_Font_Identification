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

    x= np.array(x,dtype=object)
    y= np.array(y)
    return x,y

def ReadTestSet(directory):
    x_test = []
    for filename in sorted(glob.glob(directory +'*.png')):
        img = cv2.imread(filename) 
        x_test.append(img)

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
