import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.onnx
import gurobipy as gp
from gurobipy import GRB
import torch.onnx


# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 100)
#         self.fc2 = nn.Linear(100, 20)
#         self.fc3 = nn.Linear(20, 10)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = x - 0.1
#         x = x + 0.1
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x


# model = Net()




# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 100)
#         self.fc2 = nn.Linear(100, 20)
#         self.fc3 = nn.Linear(20, 10)

#     def forward(self, x):
#         # First layer: MatMul and Add
#         x = torch.matmul(x, self.fc1.weight.t()) + self.fc1.bias
#         # Second layer: MatMul and Add
#         x = torch.matmul(x, self.fc2.weight.t()) + self.fc2.bias
#         # Third layer: MatMul and Add
#         x = torch.matmul(x, self.fc3.weight.t()) + self.fc3.bias
#         return x

# model = Net()





# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 100)
#         self.fc2 = nn.Linear(100, 20)
#         self.fc3 = nn.Linear(20, 10)

#     def forward(self, x):
#         x1 = self.fc1(x)
#         x1 = x1 - 0.1
#         x1 = x1 + 0.1
#         x2 = F.relu(self.fc2(x1))

#         # Concatenate x1 and x2 along dimension 1
#         x_concat = torch.cat((x1, x2), dim=1)

#         # Process the concatenated result through the last layer
#         x = self.fc3(x_concat[:, :20])  # Taking only the compatible input part for fc3

#         return x

# # Instantiate the model
# model = Net()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(120, 10)  # Updated input size from 20 to 120

    def forward(self, x):
        x1 = self.fc1(x)
        # x1 = x1 - 0.1
        # x1 = x1 + 0.1
        x2 = F.relu(self.fc2(x1))

        # Concatenate x1 and x2 along dimension 1
        x_concat = torch.cat((x1, x2), dim=1)  # Shape: (batch_size, 120)

        # Pass the entire concatenated tensor to fc3 without slicing
        x = self.fc3(x_concat)  # Shape: (batch_size, 10)

        return x


model = Net()

dummy_input = torch.randn(1, 28 * 28)  # Batch size of 1
# print("dummy_input:", dummy_input)
onnx_file_path = "neural_network.onnx"  # Desired output file name
torch.onnx.export(model,               # Model being exported
                  dummy_input,       # Model input (or a tuple for multiple inputs)
                  onnx_file_path,    # Where to save the model
                  export_params=True,  # Store the trained parameter weights inside the model file
                  opset_version=11,   # ONNX version to export the model to
                  do_constant_folding=True,  # Whether to execute constant folding for optimization
                  input_names=['input'],   # Name of the input layer
                  output_names=['output'],  # Name of the output layer
                  dynamic_axes={'input': {0: 'batch_size'},  # Variable-length axes
                                'output': {0: 'batch_size'}})




#REAL LIFE EXAMPLE WITH AN IMAGE#

# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(28*28, 100)
#         self.fc2 = nn.Linear(100, 50)
#         self.fc3 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # Initialize the model
# model = Net()

# # Load an MNIST image
# transform = transforms.Compose([transforms.ToTensor()])
# mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=1, shuffle=False)

# # Get a single image and its label
# images, labels = next(iter(mnist_loader))
# input_image = images.view(1, -1)  # Flatten the image to match input size
# input_label = labels.item()

# # Export the model with the real image as input
# onnx_file_path = "neural_network.onnx"  # Desired output file name
# torch.onnx.export(model,               # Model being exported
#                   input_image,         # Real image input
#                   onnx_file_path,      # Where to save the model
#                   export_params=True,  # Store the trained parameter weights inside the model file
#                   opset_version=11,    # ONNX version to export the model to
#                   do_constant_folding=True,  # Whether to execute constant folding for optimization
#                   input_names=['input'],     # Name of the input layer
#                   output_names=['output'],   # Name of the output layer
#                   dynamic_axes={'input': {0: 'batch_size'},  # Variable-length axes
#                                 'output': {0: 'batch_size'}})
