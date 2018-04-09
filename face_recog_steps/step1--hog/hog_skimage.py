import matplotlib.pyplot as plt

from skimage import color
from skimage.feature import hog
from skimage import data, exposure

from skimage import io

image = io.imread('./li_00.jpg')
image = color.rgb2gray(image)
#image = color.rgb2gray(data.astronaut())
#image = data.astronaut()

print("type of image is:{}".format(type(image)))
print("image is:{}".format(image))
print("no. of columns of image is:{}".format(len(image[0])))
print("no. of rows of image is:{}".format(len(image)))

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
