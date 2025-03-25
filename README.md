# Overview

The ONNX-To-Gurobi is a Python library that creates Gurobi models for neural networks in ONNX format.

The library has been designed to allow easy extensions, and it currently supports the following ONNX nodes:

- Add
- Sub
- MatMul
- Gemm
- ReLu
- Conv
- Unsqueeze
- MaxPool
- AveragePool
- BatchNormalization
- Flatten
- Identity
- Reshape
- Shape
- Concat
- Dropout


# Installation

We highly recommend creating a virtual conda environment and installing the library within the environment by following the following steps:

1- Gurobi is not installed automatically. Please install it manually using:
```
    conda install -c gurobi gurobi
```
2- Make sure to switch to Python 11 inside your environment using:
```
    conda install python=11
``` 

3- Install the library using:
```
    pip install onnxgurobi
```

# Getting Started

The '''ONNXToGurobi''' class provides the central interface for converting an ONNX model into a Gurobi optimization model.

To get access to the class's methods and attributes, you need to import it using:

'''
from onnx_to_gurobi.onnxToGurobi import ONNXToGurobi

'''


The '''ONNXToGurobi''' class:

- Parses the ONNX graph and constructs an internal representation of each operator and its corresponding tensor shapes.

- Creates a Gurobi model along with the necessary variables and constraints.

- Exposes all model components (decision variables, Gurobi Model object, node definitions, tensor shapes), allowing you to:

* Set or fix input variables to specific values.

* Introduce objectives.

* Add your own constraints.

* Solve the resulting MILP and then inspect or extract the outputs from the solution.


An overview of the class’s methods and attributes:

'''
class ONNXToGurobi:
    def build_model(self):
        """
        Constructs the Gurobi model by creating variables and applying operator constraints.

        """

    def get_gurobi_model(self):
        """
        Retrieves the Gurobi model after all constraints have been added.

        Returns:
            gurobipy.Model: The constructed Gurobi model reflecting the ONNX graph.
        """

    # Attributes:
    self.model               # The Gurobi Model object
    self.variables           # A dict mapping tensor names to Gurobi variables (or constants)
    self.in_out_tensors_shapes # Shapes of all input and output tensors
    self.nodes               # Node definitions parsed from ONNX
    self.initializers        # Constant tensors extracted from the ONNX graph

'''

