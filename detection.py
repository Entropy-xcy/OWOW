# -*- coding: utf-8 -*-

import cv2
import numpy as np

filename="nba.png"

print "started"

def face_detect(filename):
    face_cascade=cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
    
    img=cv2.imread(filename)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print "convert finished"
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    print "detect finished"
    i=1
    for (x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) , (x+w,y+h) , (0,255,0) , 2)
        print i
        i=i+1
    print "handling finished"
    cv2.imwrite("result.jpg", img)
    print "finished"
    
def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:
    
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
    
            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt
    
            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
    
            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
        # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
    
def drawMatchesKnn(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for match in matches:
        for mat in match:
            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx
    
            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt
    
            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
    
            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    

    # Show the image
    #cv2.imshow('Matched Features', out)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')

    # Also return the image if you'd like a copy
    return out
    
    
def orb_detect(img1,img2,num):
    orb = cv2.ORB(num, 1.2)

    # Detect keypoints of original image
    (kp1,des1) = orb.detectAndCompute(img1, None)
    
    # Detect keypoints of rotated image
    (kp2,des2) = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)
    
    return (matches,kp1,kp2)
    
