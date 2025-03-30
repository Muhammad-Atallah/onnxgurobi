import numpy as np
import pytest
from tests_utils import compare_models

@pytest.mark.integration
def test_conv1():
    """Tests a convolutional neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/conv1.onnx"
    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_conv2():
    """Tests a convolutional neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/conv2.onnx"
    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_conv3():
    """Tests a convolutional neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/conv3.onnx"
    input_data = np.random.randn(1, 1, 10, 10).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_fc1():
    """Tests a fully connected neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/fc1.onnx"
    input_data = np.random.randn(1, 28, 28).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_fc2():
    """Tests a fully connected neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/fc2.onnx"
    input_data = np.random.randn(1, 784).astype(np.float32)
    compare_models(model_path, input_data)

@pytest.mark.integration
def test_fc3():
    """Tests a fully connected neural network in ONNX format"""
    model_path = "./onnxgurobi/tests/models/fc3.onnx"
    input_data = np.random.randn(1, 784).astype(np.float32)
    compare_models(model_path, input_data)