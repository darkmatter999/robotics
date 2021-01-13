
#Offline on-device inference with custom BERT model
#More documentation on https://colab.research.google.com/drive/1DYUJXfJXMQhe1BUiurXw1Rrm8tDFC58k#scrollTo=akGTf-l_ndZy

from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification

import tensorflow as tf
import json


#tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-german-cased')

#load Tokenizer
#It is possible to save the Tokenizer locally and indicate the path in 'tokenization_distilbert.py', just to get rid of any server download

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

loaded_model = TFDistilBertForSequenceClassification.from_pretrained("sentiment_custom_model1")

test_sentence = "I like you but you're too lazy"
test_sentence_sarcasm = "News anchor hits back at viewer who sent her snarky note about ‘showing too much cleavage’ during broadcast"

# replace to test_sentence_sarcasm variable, if you want to test sarcasm
predict_input = tokenizer.encode(test_sentence,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

tf_output = loaded_model.predict(predict_input)[0]

#do stuff
print(tf_output)

#do more stuff...