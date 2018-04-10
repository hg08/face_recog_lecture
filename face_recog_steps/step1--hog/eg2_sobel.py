#!/usr/bin/env python

import numpy as np 
from matplotlib import pyplot as plt
width = 640
height = 480

Gx = np.zeros((480,640))
Gy = np.zeros((480,640))
G  = np.zeros((480,640))

pixel = np.loadtxt('img_dat.txt')
img = pixel.reshape((height,width))


for j in range(height):
	for i in range(width):
		if i == 0 or j == 0 or i == width - 1 or j == height - 1:
			Gx[j][i] = 0
			Gy[j][i] = 0
			G[j][i] = 0
		else:
			Gx[j][i] = (-1)*img[j-1][i-1]  + img[j-1][i+1] + (-2)*img[j][i-1] + \
						2*img[j][i+1] + (-1)*img[j+1][i-1] + img[j+1][i+1]
			Gy[j][i] = img[j-1][i-1] + 2*img[j-1][i] + img[j-1][i+1] + \
					(-1)*img[j+1][i-1] + (-2)*img[j+1][i] + (-1)*img[j+1][i+1]
			G[j][i] = ((Gx[j][i])**2 + (Gy[j][i])**2 )**0.5


plt.subplot(2, 2, 1)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Original')


plt.subplot(2, 2, 2)
plt.imshow(Gx, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Gx')

plt.subplot(2, 2, 3)
plt.imshow(Gy, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Gy')

plt.subplot(2, 2, 4)
plt.imshow(G, cmap=plt.cm.gray)
plt.axis('off')
plt.title('G')

plt.show()
