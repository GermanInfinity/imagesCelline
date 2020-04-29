import random, os
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage import io

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)


if __name__=="__main__":
	noise_count = 0
	rot_count = 0
	direc = '../MDA-MB-468'

	for filename in os.listdir(direc):
		if filename.endswith(".png") or filename.endswith(".jpg"):
			im = sk.io.imread(direc+'/'+filename)

			if noise_count <  24:
				transformed_img = random_noise(im)
				path = direc+'/MDA_nois(%s)_rot(%s).png'%(noise_count, rot_count)
				sk.io.imsave(path, transformed_img)
				noise_count += 1

			if  rot_count < 5:
				transformed_img = random_rotation(im)
				path = direc+'/MDA_nois(%s)_rot(%s).png'%(noise_count, rot_count)
				sk.io.imsave(path, transformed_img)
				rot_count += 1

				
			
			


		
		