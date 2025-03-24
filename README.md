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


Installation

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