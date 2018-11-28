import torch

uninitialized_tensor = torch.empty(2,2, dtype = torch.long)
print("uninitialized_tensor:", uninitialized_tensor)

initialized_tensor = torch.rand(2,2)
print("initialized_tensor", initialized_tensor)

torch_tensor = torch.tensor([5.5, 3])
print("torch tensor", torch_tensor)

torch_tensor = torch_tensor.new_ones(2,2, dtype=torch.double)
print("torch tensor with changed size", torch_tensor)

randn_like_tensor = torch.randn_like(torch_tensor, dtype=torch.float)
print("torch tensor with changes type", randn_like_tensor)

print("size of last tensor: ",randn_like_tensor.size())

temp_tens = torch.randn(2,2, dtype=torch.float)
print("added tensors", torch.add(randn_like_tensor, temp_tens))

#post fixing with _ will make the operation 'in place'.
#example

randn_like_tensor.add_(temp_tens)
print("in place addition", randn_like_tensor)

print("only first item", randn_like_tensor[:,1])

new_dim_tensor = randn_like_tensor.view(-1,1)
print("dynamic sizing in tensors", new_dim_tensor.size())

#single itemed list to single item
#example

single_len_tensor = torch.randn(1)
print("single item from tensor", single_len_tensor.item())

#converting a tensor to a numpy array
temp_torch_tensor = torch.ones(5)
print("tensor to numpy: ", temp_torch_tensor.numpy())

import numpy as np
temp_numpy_array = np.ones(5)
print("numpy to tensor: ", torch.from_numpy(temp_numpy_array))

# import numpy as np
# # X = (hours studying, hours sleeping), y = score on test, xPredicted = 4 hours studying & 8 hours sleeping (input
# # data for prediction)
# X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
# y = np.array(([92], [86], [89]), dtype=float)
# xPredicted = np.array(([4, 8]), dtype=float)
# # scale units
# X = X / np.amax(X, axis=0)  # maximum of X array
# xPredicted = xPredicted / np.amax(xPredicted, axis=0)  # maximum of xPredicted (our input data for the prediction)
# y = y / 100  # max test score is 100





