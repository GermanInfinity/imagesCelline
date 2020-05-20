import pandas as pd
import numpy as np
import time

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F


def load_images(image_size, batch_size, train_root, val_root):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = datasets.ImageFolder(root=train_root, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size+20, shuffle=True, num_workers=2)

    val_set = datasets.ImageFolder(root=val_root, transform=transform)
    val_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size-20, shuffle=True, num_workers=2)


    return val_loader, train_loader


val_loader, train_loader = load_images(224, 40, './images/train/', './images/val')




class SimpleCNN(torch.nn.Module):   
    def __init__(self):
        super(SimpleCNN, self).__init__()

        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # ((inp - kernel_size + 2*padding) / stride) + 1
        #feature_maps rom pooling layer = (32 - 2 + 0 / 2 ) + 1
        # input features = num * feature_maps
        #4608 input features
        #output features maps = (4608 - )
        #((
        #64 output features
        self.fc1 = torch.nn.Linear(2940, 72)
        
        #64 input features, 10 output features for our 10 defined classes
        #self.fc2 = torch.nn.Linear(64, 5)
        self.fc2 = torch.nn.Linear(18 * 3 * 3, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    # Defining the forward pass    
    def forward(self, x):
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 2940) #Flatten all feature maps into pytorch n rows
                                    # and specified columns

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = self.fc1(x)
        x = F.relu(x)
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        x = self.fc3(x)
        return(x)

def train(net, batch_size, n_epochs, learning_rate):
    #Get training data
    val_loader, train_loader = load_images(224, batch_size, './images/train/', './images/val')
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)



    #Get inputs
    data = next(iter(train_loader))
    inputs, labels = data

    # #Wrap them in a Variable object
    inputs, labels = Variable(inputs), Variable(labels)

    #Set the parameter gradients to zero
    optimizer.zero_grad()
    
    #Forward pass, backward pass, optimize
    outputs = net(inputs)
    print (outputs.size())
    #loss_size = loss(outputs, labels)
    #loss_size.backward()
    #optimizer.step()



def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    val_loader, train_loader = load_images(224, batch_size, './images/train/', './images/val')
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


CNN = SimpleCNN()
#trainNet(CNN, batch_size=40, n_epochs=5, learning_rate=0.001)
train(CNN, batch_size=40, n_epochs=5, learning_rate=0.001)
