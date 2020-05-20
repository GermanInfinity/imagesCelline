import os 
import numpy as np
from PIL import Image, ImageOps
import cv2

direc = "../images-aug/val/HC11/"

for filename in os.listdir(direc):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        
    	img1 = Image.open(direc+filename)
    	img1 = img1.convert('L')
    	img1.save(direc + '/' + filename)
    	y = np.expand_dims(img1, axis=-1)
    	print (y.shape)
    