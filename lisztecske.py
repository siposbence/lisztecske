# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:07:59 2017

@author: Sipos Bence
"""
#fontos dolgok beolvasása:
import cv2
import numpy as np
from matplotlib import pyplot as plt

# blob detector paraméterezése
params = cv2.SimpleBlobDetector_Params()

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.87

params.filterByArea = True
params.minArea = 15
params.maxArea = 100

#kép beolvasása
img = cv2.imread('image3.jpg')

#kép méretének kiíratása
print (img.shape)
a=np.array(img.shape)

#jobban kezelhető HSV-be alakítás
HSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

#a kozépső régió meghatározása   
x1=(a[0]//2-200)
x2=(a[0]//2+200)
y1=(a[1]//2-200)
y2=(a[1]//2+200)

# print(x1,x2,y1,y2)

#középső régió kivágása és HSV-be alakítása
center = HSV[x1:x2, y1:y2]

#a HSV-ből megtudjuk a thresholdhoz szükséges adatokat
print ("1d: ",np.median(center[:,:,0]), "2d: ",np.average(center[:,:,1]),
                        "3d: ",np.average(center[:,:,2]))
cv2.imshow('center',center)

print (center[:,:,0])

# HSV értékek a threaholdhoz
lower_center = np.array([np.median(center[:,:,0])-7,105,110])
upper_center = np.array([np.median(center[:,:,0])+7,255,255])

#színes kép thresholja:  
mask = cv2.inRange(HSV, lower_center, upper_center)
res = cv2.bitwise_and(img,img, mask= mask)

##a maszk és az eredeti kép megjelenítése    
#cv2.imshow('original',img)
#cv2.imshow('mask',mask)

# blob a fent megadott értékekkel.
detector = cv2.SimpleBlobDetector_create(params)
 
# blob-ok detektálása.
keypoints = detector.detect(mask)

# a blobok egy list-ben vannak tárolva, így a list mérete megmondja mennyi rovar van
print (len(keypoints), " db rovar van")

# Piros köröket rajzolunk a megadott helyekre
im_with_keypoints = cv2.drawKeypoints(mask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

#a jelölt kép mentése
cv2.imwrite("testblob.jpg",im_with_keypoints)

#szokásos befejezés
cv2.waitKey(0)
cv2.destroyAllWindows()

