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
#         x = x + 0.1
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
#         x = self.fc1(x)
#         x = torch.matmul(x, self.fc2.weight.t()) + self.fc2.bias
#         x = torch.matmul(x, self.fc3.weight.t()) + self.fc3.bias
#         return x

# model = Net()





########################################Conv network###################################################
#######################################################################################################
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         # Convolutional layer: 1 input channel (grayscale), 16 output channels, 3x3 kernel
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
#         # Convolutional layer: 16 input channels, 32 output channels, 3x3 kernel
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
#         # Fully connected layer: input size 32*7*7, output size 10 (e.g., for 10 classes)
#         self.fc = nn.Linear(32 * 7 * 7, 10)
#         # Max pooling layer
#     def forward(self, x):
#         # Pass input through first convolutional layer and apply ReLU
#         x = F.relu(self.conv1(x))
#         # Apply max pooling
#         x = self.pool(x)
#         # Pass input through second convolutional layer and apply ReLU
#         x = F.relu(self.conv2(x))
#         # Apply max pooling
#         x = self.pool(x)
#         # Flatten the output for the fully connected layer
#         x = torch.flatten(x, start_dim=1)
#         # Pass through fully connected layer
#         x = self.fc(x)
#         return x

# # Instantiate the model
# model = SimpleCNN()

# # Switch the model to evaluation mode
# model.eval()

# # Define a dummy input with the same shape as the model's input
# dummy_input = torch.randn(1, 1, 28, 28)  # Example: batch size = 1, channels = 1, height = 28, width = 28

# # Export the model to ONNX format
# onnx_file_path = "conv.onnx"
# torch.onnx.export(
#     model,                      # Model to export
#     dummy_input,                # Dummy input for tracing
#     onnx_file_path,             # Path to save the ONNX file
#     export_params=True,         # Store the parameters in the ONNX model
#     opset_version=11,           # ONNX version to use
#     do_constant_folding=True,   # Optimize constant folding for inference
#     input_names=['input'],      # Name of the input node
#     output_names=['output'],    # Name of the output node
#     dynamic_axes={              # Allow variable batch sizes
#         'input': {0: 'batch_size'},
#         'output': {0: 'batch_size'}
#     }
# )






##########################################################################################################
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
        self.weight = nn.Parameter(torch.randn(100, 28*28))
        self.bias = nn.Parameter(torch.randn(100))
    
    def forward(self, x):
        # Suppose you want to set alpha=2 and beta=0.5
        alpha = 2.0
        beta = 0.5
        # x here should be of shape (batch_size, 28*28)
        x = torch.addmm(self.bias * beta, x, self.weight.t(), beta=beta, alpha=alpha)
        x = x + 0.1  # The additional bias can also be thought of as a further addition.
        return x


model = Net()

dummy_input = torch.randn(1, 28 * 28)  # Batch size of 1
onnx_file_path = "simple_add.onnx"  # Desired output file name
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
