########################################################################################################################################################################
##################################################################### Loading and running a .tflite model ##############################################################
########################################################################################################################################################################

#load and conduct inferences on any pre-converted and saved .tflite file

#Can be used for any binary image classifier model in .tflite format

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys
import os

#default setting: no plot parameter input value, no plot output
plot = False

def inference(model, image_input, height, width, plotimage="no"):

    #Load example images with the Keras Image Preprocessing API
    raw_img = image.load_img(image_input, target_size=(int(height), int(width)))
    img = image.img_to_array(raw_img)
    img = np.expand_dims(img, axis=0)

    #Inference of a new image using the size-reduced .tflite model

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model) 
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # define input shape
    input_shape = input_details[0]['shape']
    #set the above defined example image array (must be of correct shape!) as input data
    input_data = img
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #infer
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.

    #An output of <0.5 means the image is classified as 'Class A', >0.5 means 'Class B'

    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_data < 0.5:
        print("The example belongs to Class A. The score is:", output_data)
    else:
        print("The example belongs to Class B. The score is:", output_data)

    if plot:
        plt.imshow(raw_img)
        plt.show()

if __name__ == "__main__":
    model = str(sys.argv[1])
    image_input = str(sys.argv[2])
    height = str(sys.argv[3])
    width = str(sys.argv[4])
    #at the end, an additional 'plot' command may be added. Then the raw input image will be plotted.
    if len(sys.argv) == 6:
        if sys.argv[5] == 'plot':
            plot = True
        else:
            pass

    inference(model, image_input, height, width)




