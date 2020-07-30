import os
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D


def get_imagelist(path):
    #loop in Python list comprehension form
    return [os.path.join(path,f) for f in os.listdir(path)]

#use the 'img/spoons' folder as path parameter and save the file path list as 'img_list'
img_list = get_imagelist('img/horse_or_human_2/horse_or_human_2_train/humans')

#print the contents of img_list
#print (img_list)

#define the uniform width and height of each sample
width = 120
height = 120

#create an empty Numpy array as a container for all images
#img_array = np.array([np.zeros((60,60,3))])
img_array = np.array([np.zeros((width, height, 3))])

#convert all images to arrays and resize them uniformly into 60 x 60 x 3 pixels, 
#flatten each array to column vector, then concatenate them to the just created new img_array
for img in img_list[0:30]:
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

#for sanity check, display second image
#conversion to uint8 format is necessary
#show_image = Image.fromarray((img_array[5]).astype(np.uint8))
#show_image.show()

input_image_shape = (120, 120, 3)
input_image = img_array[18].reshape(input_image_shape)

# Create the model
model = Sequential()
#model.add(Cropping2D(cropping=((15, 15), (30, 30)), input_shape=input_image_shape))
model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_image_shape))
#model.add(MaxPooling2D(2, 2))
#model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.summary()

# Perform actual cropping (generate a cropped image through a 'fake' prediction)
model_inputs = np.array([input_image])
outputs_cropped = model.predict(model_inputs)

# Get output
outputs_cropped = outputs_cropped[0]
print (outputs_cropped[0].shape)
print (input_image.shape)

#Image.fromarray((outputs_cropped.astype(np.uint8))).show()

# Visualize input and output
fig, axes = plt.subplots(1, 2)
#axes[0].imshow(input_image[:, :, 2]) 
axes[0].imshow(Image.fromarray((input_image.astype(np.uint8))))
axes[0].set_title('Original image')
#axes[1].imshow(outputs_cropped[:, :, 2])
axes[1].imshow(Image.fromarray((outputs_cropped.astype(np.uint8))))
axes[1].set_title('Cropped input')
fig.suptitle(f'Original and cropped input')
fig.set_size_inches(9, 5, forward=True)
plt.show()