# This example shows how to use your own input images and plot it with matplotlib.pylot

from skimage import io
import matplotlib.pyplot as plt

image = io.imread('./li_00.jpg')

print(type(image))
plt.imshow(image)
plt.show()

"""
Mostly, we won’t be using input images from the scikit-image example data sets. Those images are typically stored in JPEG or PNG format. Since scikit-image operates on NumPy arrays, any image reader library that provides arrays will do. Options include matplotlib, pillow, imageio, imread, etc.

scikit-image conveniently wraps many of these in the io submodule, and will use whatever option is available:
"""
