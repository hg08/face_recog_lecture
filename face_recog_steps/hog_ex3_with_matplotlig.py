# Matplotlib is a plotting library for Python which gives you wide variety of plotting methods.
# Here, you will learn how to display image with Matplotlib. You can zoom images, save it etc using Matplotlib.

import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('WP2.jpg',0)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
