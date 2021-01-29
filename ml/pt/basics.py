##############################################################################################################################
###########################COLLECTION OF EXAMPLES OF BASIC PYTORCH OPERATIONS#################################################
##############################################################################################################################

import numpy as np
import torch
import torchvision

'''
Basic NN training (one training example)
The gist of ML in general
'''

model = torchvision.models.resnet18(pretrained=True) #import pretrained ResNet model which will be the pre-trained model for 'fine-tuning'
data = torch.rand(1,3,64,64) #create random 'image' (3x64x64, ie 64x64 with 3 channels)
labels = torch.rand(1,1000) #create random labels (ResNet outputs 1000 features, hence the size)

#Training for 50 steps
def simple_train(model, input, labels):
    step = 0
    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) #define optimizer
    while step < 50:
        optim.zero_grad() #zeroes the gradient buffers of all parameters - without this clearing gradients will be accumulated to existing gradients
        prediction = model(input) #do forward prop, or one prediction attempt
        print (prediction)
        loss = (prediction - labels).sum() #calculate loss
        loss.backward() #initiate backword prop
        optim.step() #call optimizer to execute one SGD step
        print (f"The loss at step {step} is {int(loss)}." ) #display loss at current step
        step += 1 #increment step


'''
At times, one might want to know the total number of trainable parameters (a.k.a. weights) of a model. Here's how.
'''
def count_params(model):
    #non-trainable params do not require a gradient, so we do not count them
    #numel returns the total number of elements of an input tensor
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    return (f"The number of trainable parameters is {num_train_params}.")

'''
For reference, TF via Keras API makes it easier to see model parameters through model.summary

import tensorflow as tf
from tensorflow import keras

def tf_loadmodel():
    model = tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet', input_tensor=None,
    input_shape=None, pooling=None, classes=1000, 
    )
    print (model.summary())

'''
'''

Freeze layers / finetuning
We freeze most of the layers and only modify the classifier layers when we take a pretrained net to use on new data
'''
def finetune(model):
    #we use the above instantiated ResNet18 model again
    for param in model.parameters():
        param.requires_grad = False #freeze the whole model, i.e. take away gradient requirement resulting in no parameters being trainable anymore
    model.fc = torch.nn.Linear(512,10) #new 'dataset' which will be trained. It takes the place of the previous last linear layer
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9) #optimize only that classifier layer (based on the new data)
    count_params(model) #from the number of params we see that all parameters except the last (classifier) layer are now 'frozen'

'''
Above, we instantiated a pre-trained model which can then be used for fine-tuning. Here's how to define a model (net) from scratch
'''


import torch.nn as nn
import torch.nn.functional as F

#This model is defined in a class-based, object-oriented way
class Model(nn.Module): #write class names in uppercase

    def __init__(self):
        #main layers go here
        super(Model, self).__init__()
        # input layer --> 1 input image channel, 6 output channels, 3x3 square convolution kernel  
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    #define poolings, flattening and activations here
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    #setup flattening (flatten input array to a single number)
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
model_from_scratch = Model()

'''
define MSE loss
When calling loss, the loss grad function (grad_fn) takes into account all learnable model parameters
'''
def loss(model, input):
    output = model(input) #one prediction step, same as in 'simple_train' above
    target = torch.randn(10) #create a 'dummy' target. E.g., in a real NN this is the ground truth of an image
    target = target.view(1, -1) #reshape the dummy target to the required output format (here: (1,10))
    loss_criterion = torch.nn.MSELoss() #define which loss functions to use

    loss = loss_criterion(output, target) #measure the loss
    return (loss)

'''
do backprop and look at gradients from a particular layer
'''
def backprop(model, input):
    model.zero_grad() #set all model gradient buffers to zero
    lossf = loss(model, input)
    print('conv1.bias.grad before backward')
    print(model.conv1.bias.grad) #look at the 'conv1' layer gradients before doing backprop

    lossf.backward() #execute backprop

    print('conv1.bias.grad after backward')
    print(model.conv1.bias.grad) #look at the 'conv1' layer gradients after doing backprop

'''
weight update
After each step of backprop, the weights (aka parameters) need to be updated according to the general rule:
weight = weight - lr * gradient
'''

def update_weights(model, input):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #define optimizer (SGD in this case, with learning rate=0.01)
    #this following should be in a loop, as in 'simple_train' above
    optimizer.zero_grad()   # zero the gradient buffers
    output = model(input)
    lossf = loss(model, input)
    lossf.backward()
    print(model.conv1.weight.data) #weights before weight update
    optimizer.step()    # Does the update
    print(model.conv1.weight.data) #weights after weight update

#count_params(model)
#simple_train(model_from_scratch, torch.rand(1,1,32,32), torch.rand(1,10))
#tf_loadmodel()
#finetune(model)
#create_model()
#update_weights(model_from_scratch, torch.rand(1,1,32,32))

for param in model_from_scratch.parameters():
    print (param.numel())
print (model_from_scratch)


