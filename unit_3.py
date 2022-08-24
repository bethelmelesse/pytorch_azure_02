import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# Loading a dataset
# FashionMNIST -
# 60,000 training examples,
# 10,000 test examples
# 28*28 (h*w) grayscale image
# 10 classes

training_data = datasets.FashionMNIST(
    root="data",               # the path where the train/test data is stored
    train=True,                # specifies training or test data
    download=True,             # downloads the data from the internet if it's not available at root
    transform=ToTensor()       # specify the feature and label transformation
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# iterating and visualising the Dataset
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

'''Preparing your data for training with DataLoaders'''
# The Dataset retrieves our dataset's features and labels one sample at a time
# Features are input
# labels are output

# The Dataloader is an iterable that abstracts this complexity for us in an easy API
# data - the training data that will be used to train the model, and the test data to evaluate the model
# batch_size - the number of records to be processed in each batch
# shuffle 0- the randoms sample of the data by indices

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Iterate through the DataLoader
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Feature batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
# print(label)
# print(label.item())
plt.title(labels_map[label.item()])
plt.axis("off")
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")

# Normalization:
# is a common data pre-processing technique
# that is applied to scale or transform the data to make sure there's an equal learning contribution from each feature

# To avoid -
# a reduction of the prediction accuracy
# difficulty for the model to learn
# unfavorable distribution of the feature data ranges

# Transforms -
# used to perform some manipulation of the data and make it suitable for training
# all torchvision datasets have two parameters that accepts callables containing the transformation logic.
# 1- transform - to modify the features
# 2- torchvision.transforms - to modify the labels

# The FashionMNIST features are in PIL image format
# the labels are integers
# for training, we need the features as normalised tensors,
# and the labels as one-hot encoded tensors

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda  y: torch.zero(10,dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

# ToTensor - converts a PIL image or Numpy ndarray into a FloatTensor
# and scales the image's pizel intensity values in the range [0.,1.]

# Lambda transforms
# apply any user-defined lambda function
# It first creates a zero tensor size 10 (the number of labels in our dataset)
# and calls scatter which assigns a value=1 on the index as given by the label y

