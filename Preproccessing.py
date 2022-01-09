import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu


def preproccessing(img):
    gray = rgb2gray(img)
    th = threshold_otsu(gray)
    binary = gray > th
    if(binary[0,0] == 1):
        binary = np.bitwise_not(binary)
    
    
    return binary.astype(np.uint8)