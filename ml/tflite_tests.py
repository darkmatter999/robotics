########################################################################################################################################################################
##################################################################### Loading and running a .tflite model ##############################################################
########################################################################################################################################################################

#load and conduct inferences on any pre-converted and saved .tflite file

#Tests for horse-human and mask detection

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

#Load example images with the Keras Image Preprocessing API

#Horse-human inference examples
#first inference example (a human, Idriss Elba)
human_exp_img = image.load_img("maddinschneider.jpg", target_size=(96, 96))
human_exp = image.img_to_array(human_exp_img)
human_exp = np.expand_dims(human_exp, axis=0)

#second inference example (a horse)
horse_exp_img = image.load_img("horse_inference_example.jpg", target_size=(96, 96))
horse_exp = image.img_to_array(horse_exp_img)
horse_exp = np.expand_dims(horse_exp, axis=0)

#Mask detection inference examples
#first inference example (a horse)
mask_exp_img = image.load_img("mask_inference_example.jpg", target_size=(120, 120))
mask_exp = image.img_to_array(mask_exp_img)
mask_exp = np.expand_dims(mask_exp, axis=0)

#second inference example (a horse)
nomask_exp_img = image.load_img("nomask_inference_example.jpg", target_size=(120, 120))
nomask_exp = image.img_to_array(nomask_exp_img)
nomask_exp = np.expand_dims(nomask_exp, axis=0)

#optional plotting
#plt.imshow(human_exp_img)

#Inference of a new image using the size-reduced .tflite model

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter("horse_human_mobilenetv1_tiny.tflite") #this model 'weighs' only 45.7 KB
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
#print (input_details)
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']
#set the above defined example image array (must be of correct shape!) as input data
input_data = human_exp
interpreter.set_tensor(input_details[0]['index'], input_data)

#infer
interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.

#For horse-human, an output of <0.5 means the image is classified as a horse, >0.5 means it's a human
#For mask detection, an output of <0.5 means the image is classified as a mask-wearer, while >0.5 means that the person on the image is not wearing a mask.

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

#plt.show()
