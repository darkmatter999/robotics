
#Visualizing layer output in TensorFlow using the horse_or_human dataset as in horse_or_human_2.py

import os
import PIL
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.regularizers import l2
import random


#Directory with training horse pictures
train_horse_dir = os.path.join('img/horse_or_human_2/horse_or_human_2_train/horses')

#Directory with training human pictures
train_human_dir = os.path.join('img/horse_or_human_2/horse_or_human_2_train/humans')

#Directory with validation horse pictures
validation_horse_dir = os.path.join('img/horse_or_human_2/horse_or_human_2_val/horses')

#Directory with validation human pictures
validation_human_dir = os.path.join('img/horse_or_human_2/horse_or_human_2_val/humans')

#create a list of the filenames in each above defined subdirectory
train_horse_names = os.listdir(train_horse_dir)
print(train_horse_names[:10])

train_human_names = os.listdir(train_human_dir)
print(train_human_names[:10])

validation_horse_hames = os.listdir(validation_horse_dir)
print(validation_horse_hames[:10])

validation_human_names = os.listdir(validation_human_dir)
print(validation_human_names[:10])

#display the total number of images in each subdirectory
print('total training horse images:', len(os.listdir(train_horse_dir)))
print('total training human images:', len(os.listdir(train_human_dir)))
print('total validation horse images:', len(os.listdir(validation_horse_dir)))
print('total validation human images:', len(os.listdir(validation_human_dir)))

#display a few images using matplotlib in a 4x4 configuration

#output of images will be in a 4x4 configuration
nrows = 4
ncols = 4

#Index for iterating over images
pic_index = 0

#set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname) 
                for fname in train_horse_names[pic_index-8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname) 
                for fname in train_human_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_horse_pix+next_human_pix):
  #set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') #don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

#plt.show()

#define, compile and subsequently fit the model using the Keras ImageDataGenerator
model = tf.keras.Sequential([
    #first, one convolution layer and one max pooling layer are defined. The input shape of (60, 60, 3) is defined as 'target size' in below
    #ImageDataGenerator or in above 'manual' example
    tf.keras.layers.Cropping2D(cropping=((4, 4), (4, 4)), input_shape=(200, 200, 3)),
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(200, 200, 3)),
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

#fit a model using the Keras ImageDataGenerator (imported above)
#this simplifies the processing of the input data greatly. The above loop through the input data folder becomes redundant and all rescaling steps
#become optimized and easier to write. It takes only three lines of code to fit a pre-defined and pre-compiled sequential model.

#ImageDataGenerator takes in a directory with various (min. 2) subdirectories in which the categorized data is stored. 2 folders for binary
#classification, and k folders for k categories

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

val_dataset_IDG = ImageDataGenerator(rescale=1/255,
      #addition of various augmentation features
      #rotation_range=40,
      #width_shift_range=0.2,
      #height_shift_range=0.2,
      #shear_range=0.2,
      #zoom_range=0.2,
      #horizontal_flip=True,
      #fill_mode='nearest'
)

#load the training set
train_IDG = train_dataset_IDG.flow_from_directory('img/horse_or_human_2/horse_or_human_2_train', target_size=(200, 200), batch_size=64, class_mode='binary')
#load the validation set
val_IDG = val_dataset_IDG.flow_from_directory('img/horse_or_human_2/horse_or_human_2_val', target_size=(200, 200), batch_size=8, class_mode='binary')
#fit the model and instantiate the results as 'history'
history = model.fit(train_IDG, steps_per_epoch=8, epochs=1, verbose=1, validation_data = val_IDG, validation_steps=8)

#this is a handy way to display a complete overview of the above defined (sequential) model
#print (model.summary())

#visualize intermediate representations of individual layers as defined above
# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
#visualization_model = Model(img_input, successive_outputs)
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)
# Let's prepare a random input image from the training set.
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]
img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(200, 200))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers[1:]]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
'''
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
'''


 
