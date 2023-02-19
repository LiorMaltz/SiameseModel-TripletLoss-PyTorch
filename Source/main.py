import numpy as np
import math as m
import scipy
import matplotlib.pyplot as plt
#import pyfits
from numpy.fft import fft, fftfreq

import zipfile
import cv2

arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
       [[19, 20, 21], [22, 23, 24], [25, 26, 27]]])

arr2 = arr.swapaxes(1, 2).swapaxes(0, 2)

archive = zipfile.ZipFile("../Data/Zips/train.zip", 'r')
img_path = 'train/'
image = cv2.imread("../Data/Images/training/train/"+"42790.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, c = image.shape

#image = image.swapaxes(0,2).swapaxes(1,2)
h_new, w_new = (64, 64)
h_center, w_center = int(image.shape[0] / 2), int(image.shape[1] / 2)

result = np.empty((h, w, 3), int)
image_s = cv2.split(image)

for k in range(c):
    #plt.imshow(image_s[k])
    #plt.show()
    f = np.fft.fft2(image_s[k])
    f_shifted = np.fft.fftshift(f)
    f_new = np.zeros(f.shape, dtype=complex)

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            if abs(h_center - i) < h_new and abs(w_center - j) < w_new:
                f_new[i][j] = f_shifted[i][j]

    image_new = np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(f_new))), 0, 255)
    #plt.imshow(image_new)
    #plt.show()
    #image_new = image_new.resize((780, 540))
    #image_new = cv2.resize(image_new, (780, 540), interpolation = cv2.INTER_LINEAR)
    result[:, :, k:k+1] = np.reshape(image_new, (image_new.shape[0], image_new.shape[1], 1))
    #result[k:k+1,:,:] = np.append(result, np.array([image_new]), axis=0)

#image = image.swapaxes(1, 2).swapaxes(0, 2)
#image_new = result.swapaxes(1, 2).swapaxes(0, 2)

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()























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