import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

'''Neural Network'''

# neural network
# is a collection of neurons that are connected by layers
# each neuron is a small computing unit
# that performs simple calculations to collectively solve a problem
# they are organised in layers
# 3 types of layers - input layer, hidden layer and outter layer
# Each layers contain a number of neurons, except for the input layer

# components of a neural network
# activation function -
# determines whether a neuron should be activated or not
# if a neuron activates, then it means the input is important
# it adds non-linearity to the model

# Weights
# influence how well the output of our network will come close to the expected output value
# weights for all neurons in a layer are organised into one tensor

# bias
# makes up the difference between the activation function's output and its intended output


'''Build a neural network'''
# neural networks are comprised of layers/modules that perform operations on data
# torch.nn - provides all the building blocks you need to build your own neural network
# Every module in pytorch subclasses the nn.module
# a neural network is a module itself that consists of other modules (layers)


# to check if torch.cuda is availble, else use cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# every nn.module subclass implements the operations on input data in the forward method


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),    # First linear Module - input layer 28*28 or 784 features - takes this and transform it to a hidden layer with 512 features
            nn.ReLU(),
            nn.Linear(512, 512),   # Second linear Module - take 512 features as input from the first hidden layer and transforms it to the next hidden layer with 512 features
            nn.ReLU(),
            nn.Linear(512, 10),  # Third linear Module - take 512 features as input from the second hidden layer and transforms it to the output layer with 10
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)


# to use the model, we pass it the input data
# this executes the models forward, along with some background operations

X = torch.rand(1, 28, 28, device=device)
# print(X)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

'''Weight and bias'''
# nn.linear module randomly initializes the weight and bias for each layer

print(f"First Linear weights: {model.linear_relu_stack[0].weight} \n")
print(f"First linear bias: {model.linear_relu_stack[0].bias}\n")

# model layers - 3 images of size 28*28
input_image = torch.rand(3,28,28)
print(input_image.size())

# nn.flatten layer - to convert each 2d 28*28 image into 784 pixel
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear
# a module that applies a linear transformation on the input using it's stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU
# the ReLU activation function takes the output from the linear layer calculation and replaces the negative values with zero
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After RELU:{hidden1}\n\n")

# nn.sequential

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
print(input_image)
logits = seq_modules(input_image)
print(" ")
print(logits)

print(" ")
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
print(pred_probab)

print("model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

