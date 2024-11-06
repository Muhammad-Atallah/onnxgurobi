import onnx
import numpy as np
import struct

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
        current_shape = self.input_output_tensors_shapes.get("input")       #initializing the current shape with the input shape (will only change when going through a Gemm node)
        for node in self.graph.node:
            inputs = []
            outputs = []
            attributes = []

            # The constant node is created automatically when adding a scalar to a tensor, it has only an output and one attribute (value) in raw_data (byte)
            if node.op_type == "Constant":
                scalar_data_type = node.attribute[0].t.data_type     # Data type in form of digits
                scalar_raw_data = node.attribute[0].t.raw_data  # raw data in bytes
                for out in node.output:
                    outputs.append({'name': out, 'shape': 1})
                for attribute in node.attribute:
                    attributes.append({'name': attribute.name, 'value': struct.unpack(self._get_data_type(scalar_data_type), scalar_raw_data)[0]}) # converting bytes value based on data type

            # first input tensor, second input scalar or 1D vector, no attributes
            elif node.op_type == "Add":
                inputs.append({'name': node.input[0], 'shape': current_shape})
                if node.input[1] in self.initializer_shapes: # It's not a scalar, but a vector
                    inputs.append({'name': node.input[1], 'shape': current_shape})
                else: #It's a scalar
                    inputs.append({'name': node.input[1], 'shape': 1})
                outputs.append({'name': node.output[0], 'shape': current_shape})

            # MatMul has two inputs. The first is 1D and the second is the weights. Matrix multiplication then result in an output with 1D shape
            elif node.op_type == "MatMul":
                shape_weights = self.initializer_shapes.get(node.input[1], 1)
                shape_input = current_shape
                shape_output = shape_weights[1]
                current_shape = shape_output

                inputs.append({'name': node.input[0], 'shape': shape_input})
                inputs.append({'name': node.input[1], 'shape': shape_weights})
                outputs.append({'name': node.output[0], 'shape': shape_output})

            # 3 inputs, first is teh output of the previous node, second input is the weight, third is the bias
            # the shape of the first input is the shape of the weights (first element in a tuple of length 2), length is 2
            # the shape of the second input is teh shape of the bias, length is 1
            # the output tensor coming out of the Gemm node is now the current_shape
            elif node.op_type == "Gemm":
                shape_weights = self.initializer_shapes.get(node.input[1], 1)
                shape_bias = self.initializer_shapes.get(node.input[2], 1)
                shape_input = shape_weights[1]
                shape_output = shape_bias[0]
                current_shape = shape_output   #updating current_shape to be equal the shape of the output tensor coming out of the Gemm node

                inputs.append({'name': node.input[0], 'shape': shape_input})
                inputs.append({'name': node.input[1], 'shape': shape_weights})
                inputs.append({'name': node.input[2], 'shape': shape_bias})
                outputs.append({'name': node.output[0], 'shape': shape_output})

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
                'attributes': attributes,
                'initializers': self.initializer_values
            })
    #translating data types in numbers form into letters to be used in struct.unpack()
    def _get_data_type(self, scalar_data_type):
        if scalar_data_type == 1:  # FLOAT (32-bit float)
            return 'f'
        elif scalar_data_type == 2:  # UINT8 (8-bit unsigned integer)
            return 'B'
        elif scalar_data_type == 3:  # INT8 (8-bit signed integer)
            return 'b'
        elif scalar_data_type == 4:  # UINT16 (16-bit unsigned integer)
            return 'H'
        elif scalar_data_type == 5:  # INT16 (16-bit signed integer)
            return 'h'
        elif scalar_data_type == 6:  # INT32 (32-bit signed integer)
            return 'i'
        elif scalar_data_type == 7:  # INT64 (64-bit signed integer)
            return 'q'
        elif scalar_data_type == 10:  # FLOAT16 (16-bit float)
            return 'e'
        elif scalar_data_type == 11:  # DOUBLE (64-bit float)
            return 'd'
        elif scalar_data_type == 12:  # UINT32 (32-bit unsigned integer)
            return 'I'
        elif scalar_data_type == 13:  # UINT64 (64-bit unsigned integer)
            return 'Q'
        else:
            raise ValueError(f"Unsupported data type: {scalar_data_type}")
