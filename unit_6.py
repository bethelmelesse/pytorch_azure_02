import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

'''Optimizing the model parameters'''

# Training a model is an iterative process
# in each iteration called an epoch
# the model makes a guess about the output
# calculates the error in its guess (loss)
# collects the derivatives of the error wit respect to its parameters
# and optimizes these parameters using gradient descent

training_data = datasets.FashionMNIST(                          # done
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(                             # done
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

train_dataloader = DataLoader(training_data, batch_size=64)      # done
test_dataloader = DataLoader(test_data, batch_size=64)


class NeuralNetwork(nn.Module):                                   # done
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):                                        # done
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

'''Setting hyperparamets'''
# hyperparameters are adjustable parameters that let you control the model optimization process
# Number of Epochs - the number times the entire training dataset is pass through the network
# batch size - the number of data samples seen by the model in each epoch.
# learning rate - the size of steps the model match as it searches for best weights that will produce a higher model accuracy

learning_rate = 1e-3             # done
batch_size = 64
# epochs = 5

'''Add an optimisation loop'''
# each iteration of the optimization loop is called epoch
# each epoch consists of 2 parts: train loop and test/validation loop
# 1- the train loop - iterate over the training dataset and try to converge to optimal parameters
# 2- the validation/test loop - iterate over the test data to check if model performance is improving


# loss function
# measures the degree of dissimilarity of obtained result to the target value,
# and its is the loss function we want to minimize during training

# to calculate the loss we make a prediction using the inputs of our given data sample
# and compare it against the true data label value

loss_fn = nn.CrossEntropyLoss()                  # done

'''Optimisation pass'''
# optimisation is the process of adjusting model parameters to reduce model error in each training step

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)           # done

# inside the training loop, optimisation happens in 3 steps
# 1- call optimizer.sero_grad() to reset the gradients of model parameters.
# gradient by default add up; to prevent double-counting, we explicitly zero them at each iteration
# 2- Back-propagate the prediction loss with a call to loss.backwards()
# pyTorch deposits the gradients of the loss wrt each parameter
# 3- once we have our gradients, we call optimizer.step() to adjust the parameters by the gradients collected in the backward pass

'''Full implementation'''


def train_loop(dataloader, model, loss_fn, optimizer):               # 1- the train loop - loops over our optimization code
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:5d}]")


def test_loop(dataloader, model, loss_fn):                    # 2- the validation/test loop - evaluate model performance against our test data
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:8f} \n")


# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.prameters(), lr=learning_rate)

epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n -----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


'''Saving Models'''
torch.save(model.state_dict(), "data/model.pth")
print("Saved PyTorch Model State to model.pth")










