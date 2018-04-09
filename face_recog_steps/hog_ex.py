import matplotlib.pyplot as plt
import cv2

hog = cv2.HOGDescriptor()
im = cv2.imread('li.jpg',0)
h = hog.compute(im)
plt.plot(h)
plt.show()
