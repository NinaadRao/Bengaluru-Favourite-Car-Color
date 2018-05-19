#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:44:48 2018

@author: ninaad
"""
import cv2
import os
path = '/cars/backgroundremoval/'
files = os.listdir(path)
for j in files:
	
    src = cv2.imread(path+j, 1)
    tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(src)
    
    rgba = [b,g,r, alpha]
    dst = cv2.merge(rgba,4)
    cv2.imwrite('cars/transparent/'+j, dst)