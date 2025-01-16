# parser.py
import onnx
import numpy as np
import struct
import math
from parser_factory import ParserFactory

class ONNXParser:
    def __init__(self, onnx_model_path):
        self.onnx_model = onnx.load(onnx_model_path)
        self.graph = self.onnx_model.graph
        self.initializer_shapes = {}
        self.initializer_values = {}
        self.input_output_tensors_shapes = {}
        self.intermediate_tensors_shapes = {}
        self.constant_values = {}
        self.nodes = []
        self.current_shape = None
        self.node_parser_factory = ParserFactory()
        self._parse_model()

    def _parse_model(self):
        for initializer in self.graph.initializer:
            initializer_array = onnx.numpy_helper.to_array(initializer)
            self.initializer_shapes[initializer.name] = list(initializer_array.shape)
            self.initializer_values[initializer.name] = initializer_array

        for input in self.graph.input:
            if input.name in self.initializer_shapes:
                continue
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim[1:]]
            if len(shape) == 1:
                shape = [shape[0]]
            self.input_output_tensors_shapes[input.name] = shape

        for output in self.graph.output:
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim[1:]]
            if len(shape) == 1:
                shape = [shape[0]]
            self.input_output_tensors_shapes[output.name] = shape

        if not self.graph.input:
            raise ValueError("No input tensors found in the ONNX model.")
        self.current_shape = self.input_output_tensors_shapes[self.graph.input[0].name].copy()

        for node in self.graph.node:
            parser = self.node_parser_factory.get_parser(node.op_type)
            parser.parse(self, node, self)

    def _unsqueeze_shape(self, input_shape, axes):
        output_shape = input_shape.copy()
        for axis in sorted(axes):
            if axis < 0:
                axis += len(output_shape) + 1
            output_shape.insert(axis, 1)
        return output_shape
