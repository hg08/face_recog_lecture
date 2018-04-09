## Python gradient calculation
import cv2
import numpy as np

# Read image
im = cv2.imread('li.png')
im = np.float32(im) / 255.0

# Calculate gradient
gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)

print("gx ={}".format(gx))
print("gy ={}".format(gy))

# To use OpenCV to calculate magnitude and direction
# Python Calculate gradient magnitude and direction ( in degrees )
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

print("len(mag)={}".format(len(mag)))
print("len(angle)={}".format(len(angle)))
print("len(mag[0])={}".format(len(mag[0])))
print("len(angle[0])={}".format(len(angle[0])))
print("mag={}".format(mag))
print("angle={}".format(angle))
