#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 14:45:10 2018

@author: ninaad
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
path = 'Bengaluru-Favourite-Color-master/cars/'
files = os.listdir(path)
files=list(files)
files.sort()
#print(files)
for i in files:
    print(i)
    img = cv2.imread(path+i)
    if(str(type(img))=="<class 'numpy.ndarray'>" and img.shape[0]>500 and img.shape[1]>500):
        mask = np.zeros(img.shape[:2],np.uint8)
        
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        #print(img.shape)
        rect = (2,2,int(img.shape[0]),int(img.shape[1]))
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        img = img*mask2[:,:,np.newaxis]
        
        cv2.imwrite('Bengaluru-Favourite-Color-master/cars/backgroundremoval/'+i,img)
        