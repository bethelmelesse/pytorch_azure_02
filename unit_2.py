import torch
import numpy as np

'''Initializing Tensor'''

# directly from data

data = [[1,2],[3,4]]
x_data = torch.tensor(data)

print(f"The data:\n {data} \n")
print(f"Data to tensor:\n {x_data} \n")

# From numpy array

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

print(f"Data to numpy (np_array): \n {np_array} \n")
print(f"Numpy to Tensor (x_np): \n {x_np} \n")

np.multiply(np_array, 2, out=np_array)
print(f"Numpy np_array after * 2 operation (np_array * 2): \n {np_array}\n")
print(f"Numpy x_np value after modifying numpy array (new x_np): \n {x_np} \n")


# From another tensor

x_ones = torch.ones_like(x_data)     # retains the property of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")


# with random or constant values

shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor} \n")


# attributes of a tensor

tensor = torch.rand(3,4)
print(f"shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is store on: {tensor.device}")

# Operations on Tensors
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# standard numpy-like indexing and slicing

tensor = torch.ones(4,4)
print(f"Tensor: \n {tensor} \n")
print('First row: ', tensor[0])
print('First column: ', tensor[:,0])
print('Last column:', tensor[..., -1])
tensor[:,1] = 0
print(f"Tensor: \n {tensor} \n")


# joining tensors

t1 = torch.cat([tensor, tensor, tensor])
print(t1)


# Arithmetic operations
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)

print(f"y1: \n {y1} \n")
print(f"y2: \n {y2} \n")
print(f"y3: \n {y3} \n")

torch.matmul(tensor, tensor.T, out=y3)
print(f"y3: \n {y3} \n")

z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
print(f"z1: \n {z1} \n")
print(f"z2: \n {z2} \n")
print(f"z3: \n {z3} \n")
torch.mul(tensor, tensor, out=z3)
print(f"z3: \n {z3} \n")


# single-element tensors
print(f"Tensor: \n {tensor} \n")
agg = tensor.sum()
print(f"agg: \n {agg} \n")
agg_item = agg.item()
print(agg_item, type(agg_item))

# in place operation

tensor.add_(5)
print(f"Tensor + 5: \n {tensor} \n")

'''Bridge with Numpy'''
# Tensor to numpy array

t = torch.ones(5)
print(f"t: {t}\n")

n = t.numpy()
print(f"n: {n}\n")

t.add_(1)
print(f"t: {t}\n")
print(f"n: {n}\n")

# numpy array to tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}\n")
print(f"n: {n}\n")