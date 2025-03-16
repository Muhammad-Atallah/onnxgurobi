import numpy as np
import pytest
from tests_utils import compare_models

# @pytest.mark.integration
# def test_simple_add():
#     """Tests a small ONNX model that performs an Add operation."""
#     model_path = "tests/models/simple_ff.onnx"
#     input_data = np.random.rand(1, 28 * 28).astype(np.float32)
#     compare_models(model_path, input_data)

# @pytest.mark.integration
# def test_simple_conv():
#     """Tests a small ONNX model that performs a Conv operation."""
#     model_path = "tests/models/simple_cn.onnx"
#     input_data = np.random.rand(1, 1, 10, 10).astype(np.float32)
#     compare_models(model_path, input_data)

@pytest.mark.integration
def test_conv1():
    """Tests a convolutional neural network in ONNX format"""
    model_path = "tests/models/conv1.onnx"
    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)


@pytest.mark.integration
def test_fc1():
    """Tests a fully connected neural network in ONNX format"""
    model_path = "tests/models/fc1.onnx"
    input_data = np.random.randn(1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)
