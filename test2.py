%matplotlib inline
import scipy
import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import h5py
import time 




my_image = 'shark.jpg'
fname = './'+my_image
image = np.array(plt.imread(fname))
num_px = image.shape[1]


plt.imshow(image)
time.sleep(1000)
