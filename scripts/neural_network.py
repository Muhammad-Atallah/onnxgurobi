import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.onnx
import gurobipy as gp
from gurobipy import GRB
import torch.onnx



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Reduce channels: 1 -> 4 instead of 1->8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second conv: 4 -> 8 channels instead of 8 -> 16
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        # Batch normalization node for the 8 channels output from conv2.
        self.bn = nn.BatchNorm2d(num_features=8)
        # Average pooling node after second conv; kernel_size=2 and stride=2
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # Dropout node with dropout probability of 0.5
        self.dropout = nn.Dropout(p=0.5)
        # Flatten layer to reshape tensor from (batch_size, 8, 2, 2) to (batch_size, 32)
        self.flatten = nn.Flatten()
        # Fully connected layer with reduced input dimension: 8 * 2 * 2 = 32
        self.fc1 = nn.Linear(8 * 2 * 2, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))         # Conv1 + ReLU -> shape: (batch_size, 4, 10, 10)
        x = self.pool(x)                  # Max pooling -> shape: (batch_size, 4, 5, 5)
        x = self.conv2(x)                 # Conv2 -> shape: (batch_size, 8, 5, 5)
        x = self.bn(x)                    # Batch normalization -> shape: (batch_size, 8, 5, 5)
        x = F.relu(x)                     # ReLU activation -> shape: (batch_size, 8, 5, 5)
        x = self.avgpool(x)               # Average pooling -> shape: (batch_size, 8, 2, 2)
        x = x.unsqueeze(1)                # Unsqueeze -> shape: (batch_size, 1, 8, 2, 2)
        x = self.dropout(x)               # Dropout -> shape remains (batch_size, 1, 8, 2, 2)
        # Reshape node: remove the singleton dimension (index 1)
        x = x.reshape(1, 8, 2, 2)  # New shape: (batch_size, 8, 2, 2)
        x = self.flatten(x)               # Flatten -> shape: (batch_size, 32)
        x = self.fc1(x)                   # Fully connected layer -> shape: (batch_size, 10)
        return x

model = Net()

dummy_input = torch.randn(1, 1, 10, 10)  # Batch size of 1
onnx_file_path = "simple_cn.onnx"  # Desired output file name
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

##########################################################################################
y = torch.rand(10,1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Flatten the input.
        self.flatten = nn.Flatten()
        # A linear layer which will be exported as a Gemm node.
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)
        # Register a constant to add; this will be exported as an Add node.

    def forward(self, x):
        # Flatten the input tensor.
        x = self.flatten(x)
        # Apply the linear transformation (Gemm).
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = x + x - x
        x = torch.matmul(x, y)
        x = torch.cat([x, x], dim=1)
        return x

model = Net()

dummy_input = torch.randn(1, 784)  # Batch size of 1
onnx_file_path = "simple_ff.onnx"  # Desired output file name
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

####################################################

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # Convolution: 1 input channel (grayscale) to 16 feature maps.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # Convolution: 16 to 32 feature maps.
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # Max pooling layer.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Batch normalization for 32 feature maps.
        self.bn = nn.BatchNorm2d(32)
        # Dropout for regularization.
        self.dropout = nn.Dropout(p=0.5)
        # Flatten the output.
        self.flatten = nn.Flatten()
        # Fully connected layer: assuming input image of size 28x28, after two pooling operations, feature maps are 7x7.
        # With 32 feature maps, the flattened size is 32 * 7 * 7.
        self.fc1 = nn.Linear(32 * 7 * 7, 10)  # e.g., 10 classes for classification

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 16, 28, 28)
        x = self.pool(x)           # (batch, 16, 14, 14)
        x = F.relu(self.conv2(x))  # (batch, 32, 14, 14)
        x = self.bn(x)
        x = self.pool(x)           # (batch, 32, 7, 7)
        x = self.dropout(x)
        x = self.flatten(x)        # (batch, 32*7*7)
        x = self.fc1(x)            # (batch, 10)
        return x
    
model = Net1()

dummy_input = torch.randn(1, 1, 28, 28)  # Batch size of 1
onnx_file_path = "conv1.onnx"  # Desired output file name
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