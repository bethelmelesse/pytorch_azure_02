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
