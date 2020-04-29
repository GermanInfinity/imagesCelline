import os
from PIL import Image
import random 
direc = '../HC11'

count = 0

for filename in os.listdir(direc):
	if filename.endswith(".png") or filename.endswith(".jpg"):
		img = Image.open(direc+'/'+filename)  # Opens up selected Image
		amount = 3

		while amount > count:
			rotate = random.randint(0,360)
			output = img.rotate(rotate, expand=True)
			output.save('./filename%s.jpg'  % count)
			print ('./filename%s' % count)
			count += 1
			if count == 2: 
				break 