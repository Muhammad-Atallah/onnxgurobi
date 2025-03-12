import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.onnx
import gurobipy as gp
from gurobipy import GRB
import torch.onnx

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # Reduce channels: 1 -> 4 instead of 1->8
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         # Second conv: 4 -> 8 channels instead of 8 -> 16
#         self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
#         # Flatten layer to reshape tensor from (batch_size, 8, 5, 5) to (batch_size, 8*5*5)
#         self.flatten = nn.Flatten()
#         # Fully connected layer with reduced input dimension: 8 * 5 * 5 = 200
#         self.fc1 = nn.Linear(8 * 5 * 5, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))   # Conv1 followed by ReLU -> shape: (batch_size, 4, 10, 10)
#         x = self.pool(x)            # Max pooling -> shape: (batch_size, 4, 5, 5)
#         x = F.relu(self.conv2(x))   # Conv2 followed by ReLU -> shape: (batch_size, 8, 5, 5)
#         x = self.flatten(x)         # Flatten -> shape: (batch_size, 200)
#         x = self.fc1(x)             # Fully connected layer -> shape: (batch_size, 10)
#         return x
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
<<<<<<< Updated upstream
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 50)
        self.fc3 = nn.Linear(50, 10)
        # self.fc4 = nn.Linear(128, 128)
        # self.fc5 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc3(x)  # No activation for final layer (e.g., use Softmax externally if needed)
=======
        # Flatten the input.
        self.flatten = nn.Flatten()
        # A linear layer which will be exported as a Gemm node.
        self.fc = nn.Linear(20, 5)
        # Register a constant to add; this will be exported as an Add node.

    def forward(self, x):
        # Flatten the input tensor.
        x = self.flatten(x)
        # Apply the linear transformation (Gemm).
        x = self.fc(x)
        # Apply the ReLU activation.
        x = F.relu(x)
        # Add the constant (Add node).
        x = x + 0.1
>>>>>>> Stashed changes
        return x

model = Net()

dummy_input = torch.randn(1, 20)  # Batch size of 1
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

