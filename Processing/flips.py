import os
from PIL import Image, ImageOps

direc = "../MDA-MB-468/"

for filename in os.listdir(direc):
	if filename.endswith(".png") or filename.endswith(".jpg"):
		im = Image.open(direc+filename)
		im_flip = ImageOps.flip(im)
		im_flip.save(direc+'flip-'+filename, quality=95)
		im_mirror = ImageOps.mirror(im)
		im_mirror.save(direc+'mirror-'+filename, quality=95)