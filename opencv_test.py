#!/usr/bin/python
#coding:utf8
'for my image'
__author__ = 'Hippo'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

image = cv2.imread("test.jpg")
#cv2.imshow("Original",image)

chans = cv2.split(image)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

print "w",len(image[0]),"h",len(image),"point",len(image[0][0])
print image

img2 = image[:,:,::-1]


#im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)   #转换了灰度化

#b, g, r = cv2.split(img)
#img2 = cv2.merge(r,g,b)
#img2 = cv2.merge([chans[2],chans[1],chans[0])
#img2 = imag[:,:,::-1]    this can be faster
#plt.subplot(121);plt.imshow(img)  # expects distorted color
#plt.subplot(122);plt.imshow(img2)  # expects true color


'''
直方图均衡化 可以锐化图片，但是会丢失细节

函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围

equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵
dst：默认即可

'''

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])




plt.savefig("out.png")#显示图像


