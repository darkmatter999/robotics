# This Computer Vision dataset classifies if a given image shows a human or a horse.
# It consists of 1000 images, 500 for each category.
# The 'human' category consists of examples of the following groups (each male and female, adult and kid):
# Caucasian - Black - East Asian - South Asian
# For each of these four subcategories there is an equal amount of adult and kid images, separated between male and female.
# 512 images in total / 4 = 128 / 4 = 32 images per subcategory
# Naming: hu-cman, hu-cfan, hu-cmkn, hu-cfkn ... for black just start with 'hu-b', for East Asian with 'hu-e', for South Asian with 'hu-s'
# above, 'n' stands for the current image number, i.e. pics from 1 to n
# likewise, 'hu' stands for 'human'

# The horse category consists of one folder of examples containing images of horses in various sizes, colors and poses:
# 512 images in total 
# Naming: ho-n

import os
#import PIL
#from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2

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
    #tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    #the flatten layer learns no parameters, it just flattens image array to a single number. 
    tf.keras.layers.Flatten(),
    #one dense layer is added
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
    # in this pet problem, the output is either 0 or 1, hence binary
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

#instantiating training and validation with Keras ImageDataGenerator. Augmentation may or may not be enabled.
train_dataset_IDG = ImageDataGenerator(rescale=1/255,
      #addition of various augmentation features
      #rotation_range=40,
      #width_shift_range=0.2,
      #height_shift_range=0.2,
      #shear_range=0.2,
      #zoom_range=0.2,
      #horizontal_flip=True,
      #fill_mode='nearest'
)

val_dataset_IDG = ImageDataGenerator(rescale=1/255)

#load training and validation data
train_IDG = train_dataset_IDG.flow_from_directory('img/horse_or_human_2/horse_or_human_2_train', target_size=(120, 120), batch_size=64, class_mode='binary')
val_IDG = val_dataset_IDG.flow_from_directory('img/horse_or_human_2/horse_or_human_2_val', target_size=(120, 120), batch_size=8, class_mode='binary')

#fit the model
history = model.fit(train_IDG, steps_per_epoch=8, epochs=120, verbose=1, validation_data = val_IDG, validation_steps=8)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc, ls='--' )
plt.title ('Training and validation accuracy')
#plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
#plt.plot  ( epochs,     loss )
#plt.plot  ( epochs, val_loss )
#plt.title ('Training and validation loss'   )

plt.show()