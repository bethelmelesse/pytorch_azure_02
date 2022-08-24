import torch

# back propagation
# gradient

# the loss function
# calculates the difference between the expected output and
# the actual output that a neural network produces

# The goal is to get the result of the loss function to zero as possible

# back propagation
# the algorithm traverse backwards through the network to adjust the weights and bias to retrain the model

# This back and forward process of retraining the model overtime to reduce the loss is called gradient descent

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

print("Gradient function for z =", z.grad_fn)
print("Gradient function for loss =", loss.grad_fn)

loss.backward()
print(w.grad)
print(b.grad)

z = torch.matmul(x, w) + b
print(z.requires_grad)


'''Disabling gradient tracking'''
with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

# reason for disabling gradient
# to mark some parameters in the neural network at frozen parameters
# to speed up computations when you are only doing forward pass



