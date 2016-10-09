# -*- coding: utf-8 -*-
#this file is just to test how the sift algorithms work in detecting the robot

import cv2
from matplotlib import pyplot as plt
import detection

img1 = cv2.imread('ov.jpg',0)          # queryImage
img2 = cv2.imread('ovf.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = detection.drawMatchesKnn(img1,kp1,img2,kp2,good)

cv2.imshow('a',img3)
cv2.waitKey(0)
cv2.destroyWindow('a')
