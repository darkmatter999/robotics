
#Preparing an ML dataset for image classification

# Implement the following steps:
# **1** loop through the directory of images to obtain a list of all file paths
# **2** define the dimensions deemed required for the images to train the model on
# **3** initialize a Numpy array in which all images are stored after they are converted to arrays
# **4** iterate through each image and do the following:
#       **4.1** resize according to above defined new (smaller) dimensions. All images must have the same dimensions.
#       **4.2** convert all images to RGB unless you have solely grayscale images in the first place. This avoids a dimensional
#               mismatch if some training example images are of RGBA format.
#       **4.3** convert each image to array format and optionally flatten each array. If the latter is not done in this step,
#               it needs to be done in a separate training layer
#       **4.4** concatenate each image array to above initialized container array. This container array is the training set.
# **5** scale all color values (max 255) to be in the range from 0 to 1. This accelerates computation.
# **6** define a array of labels accompanying each image/training example

# these above six steps are preparatory. The ML workings themselves can either be coded 'from scratch' or implemented with Tensorflow or
# another ML framework

# **7** define, compile and fit the learning model (here using Tensorflow)

import os
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf

#function creating a list of multiple file paths for images located in one local directory

def get_imagelist(path):
    #loop in Python list comprehension form
    return [os.path.join(path,f) for f in os.listdir(path)]

#use the 'img/spoons' folder as path parameter and save the file path list as 'img_list'
img_list = get_imagelist('img/spoons')

#print the contents of img_list
#print (img_list)

#define the uniform width and height of each sample
width = 60
height = 60

#create an empty Numpy array as a container for all images
#img_array = np.array([np.zeros((60,60,3))])
img_array = np.array([np.zeros((width, height, 3))])

#convert all images to arrays and resize them uniformly into 60 x 60 x 3 pixels, 
#flatten each array to column vector, then concatenate them to the just created new img_array
for img in img_list:
    #on opening, resize each image
    img = Image.open(img).resize((width,height))
    #convert to RGB (3 color channels)
    img = img.convert('RGB')
    #convert image to array
    img = np.asarray(img)
    #optionally, each image array may be flattened immediately (hence the flattening layer below can be skipped)
    #img = np.reshape(np.asarray(img).flatten(), ((width*height*3, 1)))
    #concatenate each array to img_array
    img_array = np.concatenate((img_array, np.array([img])))

#print (len(img_array))
#print (img_array[1])
print (np.shape(img_array)) # in this case (4, 60, 60, 3)

#for sanity check, display second image
#conversion to uint8 format is necessary
#show_image = Image.fromarray((img_array[1] * 255).astype(np.uint8))
#show_image.show()

#in order to make computation quicker, scale all color values to be of range 0 to 1 (divide the entire array by the max of color palette, 255)
img_array = img_array / 255.0

#create an array for the labels
label_array = np.array([0,1,1,0])

#the Tensorflow input pipeline inner workings must be checked
'''
train_dataset = tf.data.Dataset.from_tensor_slices((img_array, label_array))
#print (train_dataset)
'''

'''
#define, compile and fit the model
model = tf.keras.Sequential([
    #the first layer learns no parameters, it just flattens each (60, 60, 3) image array. This computation can also be done as part of
    #the above preprocessing steps
    tf.keras.layers.Flatten(input_shape=(60, 60, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    # in this pet problem, the output is either 0 or 1, hence binary
    tf.keras.layers.Dense(2)
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

model.fit(img_array, label_array, epochs=10)
'''