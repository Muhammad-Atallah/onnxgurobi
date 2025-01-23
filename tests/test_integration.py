import pytest
import numpy as np
import onnxruntime as ort
from gurobipy import GRB
from tests_utils import compare_models

# Make sure you import the classes from your code base correctly.
# For example, if your library's entry point is ONNXToGurobi in onnx_to_gurobi.py, do:
# from your_library_name.onnx_to_gurobi import ONNXToGurobi

@pytest.mark.integration
def test_simple_add():
    """Tests a small ONNX model that performs an Add operation."""
    model_path = "tests/models/simple_add.onnx"
    input_data = np.random.rand(1, 28*28).astype(np.float32)  # Example shape
    compare_models(model_path, input_data)

# def test_sub_operation(self):
#     """
#     Test a small ONNX model that performs a Sub operation.
#     """
#     model_path = "tests/models/simple_sub.onnx"
#     input_data = np.random.rand(1, 3).astype(np.float32)
#     compare_models(model_path, input_data)

# def test_matmul_operation(self):
#     """
#     Test a small ONNX model that performs a MatMul operation.
#     """
#     model_path = "tests/models/simple_matmul.onnx"
#     # For a MatMul, inputs might be 2D: e.g. (1, 4) times (4, 3) -> (1, 3)
#     input_data = np.random.rand(1, 4).astype(np.float32)
#     compare_models(model_path, input_data)

# def test_conv_operation(self):
#     """
#     Test a small ONNX model that performs a 2D Convolution.
#     """
#     model_path = "tests/models/simple_conv.onnx"
#     # Suppose the input is shape (1, 3, 8, 8) if there's a batch dimension
#     # or just (3, 8, 8) if the library is ignoring batch size in shapes
#     input_data = np.random.rand(1, 3, 8, 8).astype(np.float32)
#     compare_models(model_path, input_data)

# def test_concat_operation(self):
#     """
#     Test a small ONNX model that performs a Concat operation on two tensors.
#     """
#     model_path = "tests/models/simple_concat.onnx"
#     # Suppose it has 2 inputs of shape (1, 3) each. 
#     # In that case, we'll pass a dict for multiple inputs:
#     input_data = {
#         "inputA": np.random.rand(1, 3).astype(np.float32),
#         "inputB": np.random.rand(1, 3).astype(np.float32)
#     }
#     compare_models(model_path, input_data)

# # Add more tests as needed (Gemm, Flatten, Reshape, etc.).
