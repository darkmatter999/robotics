
#***LOADING THE PREVIOUSLY SAVED HORSE OR HUMAN DATASET WITH KERAS AND INFER, SUBSEQUENTLY CONVERT MODEL INTO .tflite AND SAVE .tflite MODEL***

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import matplotlib.pyplot as plt

def get_imagelist(path):
    #loop in Python list comprehension form
    return [os.path.join(path,f) for f in os.listdir(path)]

#this following would be the model call if the model was saved with tf.SavedModel
#saved_model = tf.saved_model.load('saved_horse_human3')

#***Here we load the model with the Keras API so that it is easier to instantiate it (and call model.predict) afterwards***

#the model was previously saved to disk using the Keras model saving function
#it is also possible to load via Keras a model which was initially saved by tf.SavedModel (provided the model is in the .pb format and not in .h5 format)
#saved_model = keras.models.load_model('saved_horse_human3')

#convert the saved model in .tflite format and reduce size (here from 215 KB to 149 KB) through quantization and pruning.
converter = tf.lite.TFLiteConverter.from_saved_model("saved_horse_human_with_q_aware_training") 
#***test further optimizations in .tflite models***
#the default optimizer reduces the model a further > 100 KB to only 45.7 KB
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#further optimizations (representative data, explicit quantization) for further study
'''
def representative_data_gen(): 
    for input_value, _ in test_batches.take(50):
        yield [input_value]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
'''
tflite_model = converter.convert()

#save the converted .tflite model
with open('model_qat.tflite', 'wb') as f:
    f.write(tflite_model)

#in this way, the model is already instantiated and can thus be called without further processing
#This means that the loaded model can be applied to any new inference, the weights are now 'fixed', or, the model is deployable
'''
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
        classes = saved_model.predict(images, batch_size=10)
        #print(classes.ndim)
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

unseen_correct = predict_image_batch('img/horse_or_human_unseen_images')
'''
