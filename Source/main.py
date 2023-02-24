import numpy as np
import math as m
import scipy
import matplotlib.pyplot as plt
#import pyfits
from numpy.fft import fft, fftfreq
import time
import zipfile
import cv2
from tqdm import tqdm

epochs = [1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6]

for i in tqdm(range(10000), desc="epochs", colour="green"):
    time.sleep(0.1)

























#plt.figure()
#image2 = cv2.imread("../Data/Images/training/train/"+"75861.jpg")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#image = 255 - image[:,:]
#image2 = 255 - image2[:,:]
##print(imgdata)
#plt.subplot(1, 2, 1)
#plt.imshow(image)
#plt.subplot(1, 2, 2)
#plt.imshow(image2)
#plt.show()