# -*- coding: utf-8 -*-

import cv2
import detection

img1=cv2.imread('ov.jpg',0)
img2=cv2.imread('ovf.jpg',0)

matches,kp1,kp2=detection.orb_detect(img1,img2,60)

img=detection.drawMatches(img1,kp1,img2,kp2,matches)

cv2.imshow('a',img)
cv2.waitKey(0)
cv2.destroyWindow('a')