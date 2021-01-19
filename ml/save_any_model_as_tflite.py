
#***LOAD ANY KERAS/TF MODEL, CONVERT MODEL INTO .tflite AND SAVE .tflite MODEL***

import tensorflow as tf
from tensorflow import keras

def convert_to_tflite(orig_model, tflite_model_name):

    #convert the saved model in .tflite format and reduce size (here from 215 KB to 149 KB) through quantization and pruning.
    converter = tf.lite.TFLiteConverter.from_saved_model(orig_model) 
    #implement further optimizations that reduce the model size even further
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
    with open(tflite_model_name, 'wb') as f:
        f.write(tflite_model)

convert_to_tflite("horse_human_mobilenetv1", "horse_human_mobilenetv1_tiny.tflite")