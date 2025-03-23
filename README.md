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

Gurobi is not installed automatically. Please install it manually using:
    conda install -c gurobi gurobi

Afterward run the following:
    pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple onnxgurobi