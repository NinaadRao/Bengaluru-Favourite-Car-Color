# Bengaluru-Favourite-Car-Color
## The steps in running this code is:
1. first run the deep_learning_object_detection.py. make sure all your path names for the corresponding models of car, caffe model file path and the prototxt file path names are put properly.
2. After executing the above code, You will get a folder named cars inside the directory where the above code exists. This folder contains all the cars that were detected by the dnn model.
3. Our next task is to try and remove the background of the images and for this, we use the grabcut algorithm and the code for this is named as GrabCutAlgorithm.py. Make sure you have renamed your directories and the car images for which you are supposed to execute this code was for the output of the previous code i.e from the directory where you have the images of the detected cars.
4. Now that you have executed the sentdextrans.py code, you can see that there is another folder called backgroundremoval inside the folder car, Now its time to execute the file backgroundmadetransparent.py. This code removes the background and makes it transparent. The output of this code is available in another folder named "transparent" which is again inside the folder "cars".
5. After this it's the last step. Its time to execute the kmeansclustering.py. Your filepaths should be changed according to your needs but the path to be considered is both "cars" folder and 'transparent' folder. Be patient as it takes a while to execute completely. After it has executed, a csv file is created named filedetails.csv.
6. Open this file. It has four columns. The first column has the details of the file. The next three columns are the r,g,b values respectively which dominated the car color. The last column is not right but its just for reference. Once we have the range of values for different colours, use these values and find the color it belongs to. 

## Done by
Abhishikta Sai <br/>
Niharika Gurram <br/>
Ninaad R. Rao <br/>
## Algorithms Used:
<ol>
<li>To detect the car from Bengaluru traffic- A caffe implementation of MobileNet-SSD detection network, with pre trained weights was used for detecting car which uses Deep Neural Network.
  </li><li>Background Removal- To remove the background from the cropped detected car image, Grab Cut Algorithm was used so that only the body of the car is considered.
</li><li>Color Detection- For the color detection, opencv was used and to get the most dominant color, Kmeans Algorithm was used
</li><li>Map-Reduce- After getting the output i.e. color of the car in a text file along with the file name, Hadoop Map-Reduce was used to get the number of cars of different colors.
</li><li>A manual check was done for each image to check what color a human would tell and what the result of this project yielded and a graph was drawn based on the expected vs the actual result.
  </li></ol>
