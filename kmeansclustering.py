#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 11:21:02 2018

@author: ninaad
"""

# import the necessary packages
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
# construct the argument parser and parse the arguments
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

lower_brown = np.uint8([10,30,50])
upper_brown = np.uint8([25,220,220])
	
lower_white = np.uint8([0,0,200])
upper_white = np.uint8([180,10,255])
	
lower_black = np.uint8([0,0,0])
upper_black = np.uint8([180,255,40])
	
lower_grey = np.uint8([0,0,50])
upper_grey = np.uint8([180,75,120])

lower_s = np.uint8([0,0,100])
upper_s = np.uint8([180,30,200])
	
lower_green = np.uint8([41,30,50])
upper_green = np.uint8([80,255,255])

lower_blue = np.uint8([90,30,50])
upper_blue = np.uint8([140,255,255])
	
lower_red1 = np.uint8([0,30,50])
upper_red1 = np.uint8([10,255,255])

lower_red2 = np.uint8([160,30,50])
upper_red2 = np.uint8([179,255,255])

lower_orange = np.uint8([10,30,50])
upper_orange = np.uint8([25,255,255])

lower_yellow = np.uint8([25,30,50])
upper_yellow = np.uint8([40,255,255])

lower_pink = np.uint8([141,30,50])
upper_pink = np.uint8([159,255,255])

def getcolor(rgb):
    p=np.uint8([[rgb]])
    hsvp = cv2.cvtColor(p,cv2.COLOR_BGR2HSV)
    col = hsvp[0][0]
    
    if cv2.inRange(hsvp,lower_green,upper_green):
        return "green"
    if cv2.inRange(hsvp,lower_blue,upper_blue):
        return "blue"
    if cv2.inRange(hsvp,lower_brown,upper_brown):
        return "brown"
    if cv2.inRange(hsvp,lower_orange,upper_orange):
        return "orange"
    if cv2.inRange(hsvp,lower_yellow,upper_yellow):
        return "yellow"
    if (cv2.inRange(hsvp,lower_red1,upper_red1)) or (cv2.inRange(hsvp,lower_red2,upper_red2))  :
        return "red"
    if cv2.inRange(hsvp,lower_pink,upper_pink):
        return "pink"
    if cv2.inRange(hsvp,lower_white,upper_white):
        return "white"
    if cv2.inRange(hsvp,lower_black,upper_black):
        return "black"
    if cv2.inRange(hsvp,lower_grey,upper_grey):
        return "grey"
    if cv2.inRange(hsvp,lower_s,upper_s):
        return "silver"
    else:
        return "grey"

path = 'Bengaluru-Favourite-Color-master/cars/transparent/'
files = os.listdir(path)
path1='Bengaluru-Favourite-Color-master/cars'
files1=os.listdir(path1)
import webcolors
'''
def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name
'''
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()
 
	# return the histogram
	#print(numLabels)
    
	return (hist,numLabels)

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
	list_of_colors=[]
	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		#print(color.astype("uint8").tolist())
		list_of_colors.append(color.astype("uint8").tolist())
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	#print((bar))
	# return the bar chart
	return bar,list_of_colors

file_name=[]
r_value=[]
g_value=[]
b_value=[]
color_value=[]
count=0
files1.sort()
# load the image and convert it from BGR to RGB so that
# we can dispaly it with matplotlib
for i in files1:
    image = cv2.imread(path1+i)
    #image=cv2.imread('/home/ninaad/object-detection-deep-learning/cars/cars124-0.png')
    if(str(type(image))=="<class 'numpy.ndarray'>"):
        count+=1
        print(i,count)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(type(image))
        file_name.append(i)
        # show our image
        plt.figure()
        plt.axis("off")
        plt.imshow(image)
        # reshape the image to be a list of pixels
        hei=image.shape[0]
        wid=image.shape[1]
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        # cluster the pixel intensities
        clt = KMeans(n_clusters = 4)
        clt.fit(image)
        
        
         
        
        # build a histogram of clusters and then create a figure
        # representing the number of pixels labeled to each color
        hist,numLabels = centroid_histogram(clt)
        hist=list(hist)
        #print(hist)
        import copy
        hist2=copy.deepcopy(hist)
        hist2.sort()
        
        numLabels=list(numLabels)
        
        
        bar,list_of_colors = plot_colors(hist, clt.cluster_centers_)
        list_of_colors=list(list_of_colors)
        requested_colour=[]
        actual_name=''
        closest_name=''
        if(hei<500 or wid<500):
            #print('hello_world')
            #print(hist2)
            hist2.sort()
            #print(hist2)
            #print(hist[-2])
            requested_colour = list_of_colors[hist.index(hist2[-1])]
            requested_colour=requested_colour[::-1]
            closest_name=getcolor(requested_colour)
            #print(closest_name)
            if(('black' in closest_name or 'grey' in closest_name)):
                #print('small and black or grey is the primary colour ',closest_name)
                requested_colour = list_of_colors[hist.index(hist2[-2])]
                requested_colour=requested_colour[::-1]
                closest_name=getcolor(requested_colour)
                
            #print(requested_colour)
        else:
            hist2.sort()
            requested_colour = list_of_colors[hist.index(hist2[-1])]
            requested_colour=requested_colour[::-1]
            closest_name = getcolor(requested_colour)
        if(('black' in closest_name or 'grey' in closest_name or 'silver' in closest_name or 'brown' not in closest_name) and (i in files)):
            image = cv2.imread(path+i)
            print('hello_world')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
         
            # show our image
            plt.figure()
            plt.axis("off")
            plt.imshow(image)
        # reshape the image to be a list of pixels
            image = image.reshape((image.shape[0] * image.shape[1], 3))
        # cluster the pixel intensities
            clt = KMeans(n_clusters = 4)
            clt.fit(image)
            hist,numLabels = centroid_histogram(clt)
            hist=list(hist)
            print(hist)
            import copy
            hist2=copy.deepcopy(hist)
            hist2.sort()
            numLabels=list(numLabels)
            
            
            bar,list_of_colors = plot_colors(hist, clt.cluster_centers_)
            list_of_colors=list(list_of_colors)
            requested_colour1 = list_of_colors[hist.index(hist2[-2])]
            requested_colour1=requested_colour1[::-1]
            closest_name1 = getcolor(requested_colour1)
            #print('actual name',actual_name,'closest_name',closest_name1)
            if('black' not in closest_name1 or 'grey' not in closest_name1):
                closest_name=closest_name1
                requested_colour=requested_colour1
            
        # show our color bart
        
        requested_colour=list(requested_colour)
        b_value.append(requested_colour[0])
        g_value.append(requested_colour[1])
        r_value.append(requested_colour[2])
        color_value.append(closest_name)
        plt.figure()
        plt.imshow(bar)
        plt.show()
        print('closest_name: ',closest_name,requested_colour)
import pandas as pd
df=pd.DataFrame({'r_value':r_value,'g_value':g_value,'filename':file_name,'b_value':b_value,'color_name':color_value})
df.to_csv('Bengaluru-Favourite-Color-master/filesdetails.csv',index=False)   