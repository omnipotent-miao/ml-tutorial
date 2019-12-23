import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# The images are 4 bit, 8 bit and 32 bit, respectively.
DIR = 'C:\\Users\\JoeMeli\\Documents\\git\\imageqc\\'
images = ['001.png', '100.png', '735.png']

for i in images:
    img = plt.imread(DIR + i)#[:,:,:3]
    imgarr = np.array(img)
    print(imgarr.min(), imgarr.max())
    print(imgarr.shape)

imgplot = plt.imshow(plt.imread(DIR + '735.png')[:,:,:3])
plt.show()
