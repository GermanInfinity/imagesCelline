from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import numpy as np
from PIL import Image

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
#X = datagen.flow_from_directory('../HC11/',class_mode='binary',batch_size=17)
im = np.array(Image.open('../HC11/HC11(1).jpg').resize((224,224)))

#im = im.astype('float32')
datagen.fit(im)
#Image.fromarray(im).save('../HC11/a.jpg')

