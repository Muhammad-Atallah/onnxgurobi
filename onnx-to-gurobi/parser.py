import onnx
import numpy as np

class ONNXParser:
    def __init__(self, onnx_model_path):
        self.onnx_model = onnx.load(onnx_model_path)
        self.graph = self.onnx_model.graph
        self.initializer_shapes = {}
        self.initializer_values = {}
        self.input_output_tensors_shapes = {}
        self.nodes = []
        self._parse_model()

    def _parse_model(self):
        # Fill initializer_shapes and initializer_values
        for initializer in self.graph.initializer:
            self.initializer_shapes[initializer.name] = onnx.numpy_helper.to_array(initializer).shape
            self.initializer_values[initializer.name] = onnx.numpy_helper.to_array(initializer)

        # Fill input_output_tensors_shapes
        for input in self.graph.input:
            self.input_output_tensors_shapes[input.name] = input.type.tensor_type.shape.dim[1].dim_value

        for output in self.graph.output:
            self.input_output_tensors_shapes[output.name] = output.type.tensor_type.shape.dim[1].dim_value

        # Form node objects
        for node in self.graph.node:
            inputs = []
            for inp in node.input:
                shape = self.initializer_shapes.get(inp, self.input_output_tensors_shapes.get(inp, 1))
                inputs.append({'name': inp, 'shape': shape})
            outputs = []
            for out in node.output:
                shape = self.initializer_shapes.get(out, self.input_output_tensors_shapes.get(out, 1))
                outputs.append({'name': out, 'shape': shape})
            attributes = []
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.FLOAT:
                    value = attribute.f
                elif attribute.type == onnx.AttributeProto.INT:
                    value = attribute.i
                else:
                    value = None
                attributes.append({'name': attribute.name, 'value': value})
            self.nodes.append({
                'name': node.name,
                'type': node.op_type,
                'input': inputs,
                'output': outputs,
                'attributes': attributes
            })
