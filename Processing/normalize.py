import os 
import cv2 as cv
import numpy as np

direc = "../MDA-MB-468/"

for filename in os.listdir(direc):
	if filename.endswith(".png") or filename.endswith(".jpg"):
		path = r""+direc+filename
		img = cv.imread(path)
		normalizedImg = np.zeros((800, 800))
		normalizedImg = cv.normalize(img,  normalizedImg, 0, 255, cv.NORM_MINMAX)
		cv.imwrite("./"+filename, normalizedImg)



