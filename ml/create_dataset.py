
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

# !!!!!                                                                                           # !!!!!
# !!!!!    FOR NEITHER DATASET WORKED ON IN THIS NOTEBOOK THERE IS A VALIDATION SET IMPLEMENTED     !!!!!
# !!!!!                                                                                           # !!!!!

# **7** define, compile and fit the learning model (here using Tensorflow)

import os
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

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

#for sanity check, display second image
#conversion to uint8 format is necessary
#show_image = Image.fromarray((img_array[1] * 255).astype(np.uint8))
#show_image.show()

#in order to make computation quicker, scale all color values to be of range 0 to 1 (divide the entire array by the max of color palette, i.e. 255)
img_array = img_array / 255.0

#create an array for the labels
label_array = np.array([0,1,1,0])

#print (np.shape(img_array)) # in this case (4, 60, 60, 3)
#print (np.shape(label_array)) 

#the Tensorflow input pipeline inner workings must be checked
#to make it work, it requires batching and shuffling of the data, as shown below
'''
train_dataset = tf.data.Dataset.from_tensor_slices((img_array, label_array))
print (train_dataset)

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
#test_dataset = test_dataset.batch(BATCH_SIZE)
#print (train_dataset)
'''

#define, compile and subsequently fit the model using the Keras ImageDataGenerator
model = tf.keras.Sequential([
    #first, one convolution layer and one max pooling layer are defined. The input shape of (60, 60, 3) is defined as 'target size' in below
    #ImageDataGenerator or in above 'manual' example
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(120, 120, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #adding another conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #adding yet another (the third) conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #adding yet another (the fourth) conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #adding yet another (the fifth) conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #the flatten layer learns no parameters, it just flattens image array to a single number. This computation can also be done as part of
    #the above preprocessing steps
    tf.keras.layers.Flatten(),
    #one dense layer is added
    tf.keras.layers.Dense(512, activation='relu'),
    # in this pet problem, the output is either 0 or 1, hence binary
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

'''
#model.fit(img_array, label_array, epochs=10)
model.fit(train_dataset, epochs=10)
'''

#fit a model using the Keras ImageDataGenerator (imported above)
#this simplifies the processing of the input data greatly. The above loop through the input data folder becomes redundant and all rescaling steps
#become optimized and easier to write. It takes only three lines of code to fit a pre-defined and pre-compiled sequential model.

#ImageDataGenerator takes in a directory with various (min. 2) subdirectories in which the categorized data is stored. 2 folders for binary
#classification, and k folders for k categories

train_dataset_IDG = ImageDataGenerator(rescale=1/255,
      #addition of various augmentation features
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
#loading the fork_or_spoon pet dataset
#train_IDG = train_dataset_IDG.flow_from_directory('img/fork_or_spoon/', target_size=(60, 60), batch_size=64, class_mode='binary')
#loading the horse_or_human dataset from Laurence Moroney
train_IDG = train_dataset_IDG.flow_from_directory('img/horse_or_human/', target_size=(120, 120), batch_size=64, class_mode='binary')
model.fit(train_IDG, steps_per_epoch=8, epochs=15, verbose=1)

#this is a handy way to display a complete overview of the above defined (sequential) model
#print (model.summary())

#predict an unseen image
def predict_single_image(path):
    #using the Keras image loader
    img = image.load_img(path, target_size=(60, 60))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img_class = model.predict(img)
    print(img_class)
    if img_class[0]>0.5:
        #print("It is a spoon")
        print("It is a human")
    else:
        #print("It is a fork")
        print("It is a horse")

#predict multiple images (making use of above get_imagelist function)
def predict_image_batch(folder_path):
    img_list = get_imagelist(folder_path)
    correct = 0
    for path in img_list:
        #using the Keras image loader
        img = image.load_img(path, target_size=(120, 120))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        #print(classes)
        if classes[0]>0.5:
            print(path + " is a human")
            if path[33:35] == 'hu':
                correct = correct + 1
        else:
            print(path + " is a horse")
            if path[33:35] == 'ho':
                correct = correct + 1
    print('Ratio of correctly classified images: ' + str(correct/len(img_list)))

#predict_single_image('img/new_spoon1.jpg')
predict_image_batch('img/horse_or_human_unseen_images')


 
