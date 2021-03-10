##################################################################################################################################
######################################NN TRAINING IN PYTORCH######################################################################
##################################################################################################################################

import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from lenet import Model
import matplotlib.pyplot as plt

'''
modify an imported model

class modLeNet(Model):
    def __init__(self):
        super(modLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, 3) #modify the conv shape of the conv1 layer, just as an example
        
print (modLeNet())
'''

#model = Model() #instantiate the model imported above

batch_size_train = 4 
batch_size_test = 4

#DataLoader returns an iterable
#download=True only if the dataset has not already been downloaded
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='./data/', train=True, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
                             
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.CIFAR10(root='./data/', train=False, download=False,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),
  batch_size=batch_size_test, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for data in train_loader:
    print (data[0].shape)

'''
def imshow(img):
    print (img.shape)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print (img.shape)
    print (npimg.shape)
    print (np.transpose(npimg, (1,2,0)).shape)
    plt.imshow(np.transpose(img, (1, 2, 0))) #for plotting, reshape from (3,36,138) to (36,138,3)
    #plt.imshow(npimg)
    plt.show()

# get some random training images
dataiter = iter(test_loader)
images, labels = dataiter.next() #get the first iterated batch

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#define NN (or modify imported class, as sketched above)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#define loss function and optimizer

import torch.optim as optim

#loss=CrossEntropy, optim=stochastic gradient descent
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#Training
for epoch in range(2):  # loop over the dataset for 2 epochs

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
'''