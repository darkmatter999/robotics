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
import timeit

#function creating a list of multiple file paths for images located in one local directory

def get_imagelist(path):
    #loop in Python list comprehension form
    return [os.path.join(path,f) for f in os.listdir(path)]

#define, compile and subsequently fit the model using the Keras ImageDataGenerator
model = tf.keras.Sequential([
    #first, one convolution layer and one max pooling layer are defined. The input shape of (60, 60, 3) is defined as 'target size' in below
    #ImageDataGenerator or in above 'manual' example
    #if a cropping layer is prepended, 'padding=same' must be activated for the convolutions!
    #The cropping layer is an experimental setup which assumes that most of the 'objects of interest' (horses, humans) are placed rather in the
    #center of the image. Hence, cutting out the margins may lead to a learning concentration of features in the faces and bodies of the horses
    #and humans, and not so much on the backgrounds (grass, beach, mountain, city, etc.)
    #tf.keras.layers.Cropping2D(cropping=((15, 15), (30, 30)), input_shape=(120, 120, 3)),
    #Important: Padding = same must be activated for the conv layers if the network starts with a cropping layer!
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(120, 120, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    #adding another conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #adding yet another (the third) conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #adding a Dropout layer
    tf.keras.layers.Dropout(0.25),
    #adding yet another (the fourth) conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #adding yet another (the fifth) conv + pooling layer
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #adding yet another (the sixth) conv layer
    #tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    #adding yet another (the seventh) conv layer
    #tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    #the flatten layer learns no parameters, it just flattens image array to a single number. 
    tf.keras.layers.Flatten(),
    #four dense layers are added
    #tf.keras.layers.Dense(512, activation='relu', kernel_regularizer='l2'),
    tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dense(128, activation='relu'),
    #tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dense(32, activation='relu'),
    # in this pet problem, the output is either 0 or 1, hence binary
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()
lr = 0.0007
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=lr),
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
train_IDG = train_dataset_IDG.flow_from_directory('img/horse_or_human_2/horse_or_human_2_train', target_size=(120, 120), batch_size=16, class_mode='binary')
#train_IDG = train_dataset_IDG.flow_from_directory('img/horse_or_human_reduced/horse_or_human_reduced_train', target_size=(120, 120), batch_size=64, class_mode='binary')
val_IDG = val_dataset_IDG.flow_from_directory('img/horse_or_human_2/horse_or_human_2_val', target_size=(120, 120), batch_size=4, class_mode='binary')
#val_IDG = val_dataset_IDG.flow_from_directory('img/horse_or_human_reduced/horse_or_human_reduced_val', target_size=(120, 120), batch_size=8, class_mode='binary')

#implement timer (set start time) to time the model fitting
start_time = timeit.default_timer()

#fit the model
history = model.fit(train_IDG, steps_per_epoch=16, epochs=100, verbose=1, validation_data = val_IDG, validation_steps=16)

algorithm_running_time = (timeit.default_timer() - start_time) / 60
print("The time taken for model fitting is :", algorithm_running_time, "minutes")

#In order to save a model for future use (offline inference etc.) we can use either TF itself (tf.SavedModel) or Keras via tf. 'model.save' below is the Keras version.

#horse_or_human = "exp_saved_model_horse_human2"
#tf.saved_model.save(model, horse_or_human)

model.save("saved_horse_human4")


#predict some new examples
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
        print(classes)

        if classes[0]>0.5:
            print(path + " is a human")
            if path[33:35] == 'hu':
                correct = correct + 1
        else:
            print(path + " is a horse")
            if path[33:35] == 'ho':
                correct = correct + 1
    unseen_correct = str(correct/len(img_list)*100)
    print('Ratio of correctly classified images: ' + unseen_correct)
    return (unseen_correct)

#predict_single_image('img/new_spoon1.jpg')
unseen_correct = predict_image_batch('img/horse_or_human_unseen_images')

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
plt.title ('Running Time: ' + str(algorithm_running_time) + ' minutes' + '\n' + 'Correctly classified new images: ' + unseen_correct + '%' + 
'\n' + 'Training and validation accuracy -- Learning Rate: ' + str(lr))
#plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
#plt.plot  ( epochs,     loss )
#plt.plot  ( epochs, val_loss )
#plt.title ('Training and validation loss'   )

plt.show()