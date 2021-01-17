'''
import tensorflow as tf
from tensorflow import keras


#convert the saved model in .tflite format and reduce size (here from 215 KB to 149 KB) through quantization and pruning.
converter = tf.lite.TFLiteConverter.from_keras_model("sentiment_custom_model1") 

converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]
#***test further optimizations in .tflite models***
#the default optimizer reduces the model a further > 100 KB to only 45.7 KB
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#further optimizations (representative data, explicit quantization) for further study

def representative_data_gen(): 
    for input_value, _ in test_batches.take(50):
        yield [input_value]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

tflite_model = converter.convert()

#save the converted .tflite model
with open('sentiment.tflite', 'wb') as f:
    f.write(tflite_model)

'''

import tensorflow as tf
from transformers import TFDistilBertForQuestionAnswering

model = TFDistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

input_spec = tf.TensorSpec([1, 384], tf.int32)
model._set_inputs(input_spec, training=False)

print(model.inputs)
print(model.outputs)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# For normal conversion:
converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]

# For conversion with FP16 quantization:
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.target_spec.supported_types = [tf.float16]
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.experimental_new_converter = True

# For conversion with hybrid quantization:
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
# converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
# converter.experimental_new_converter = True

tflite_model = converter.convert()

open("distilbert-squad-384.tflite", "wb").write(tflite_model)