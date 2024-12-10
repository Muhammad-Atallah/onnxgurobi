# parser.py

import onnx
import numpy as np
import struct
import math

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
        self._parse_model()

    def _parse_model(self):
        # Fill initializer_shapes and initializer_values
        for initializer in self.graph.initializer:
            initializer_array = onnx.numpy_helper.to_array(initializer)
            self.initializer_shapes[initializer.name] = list(initializer_array.shape)
            self.initializer_values[initializer.name] = initializer_array

        # Fill input_output_tensors_shapes for actual inputs (exclude initializers)
        for input in self.graph.input:
            if input.name in set(self.initializer_shapes.keys()):
                continue  # Skip initializers
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim[1:]]  # Exclude batch size
            # Ensure shape is a list even if single dimensional
            if len(shape) == 1:
                shape = [shape[0]]
            self.input_output_tensors_shapes[input.name] = shape

        for output in self.graph.output:
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim[1:]]
            # Ensure shape is a list even if single dimensional
            if len(shape) == 1:
                shape = [shape[0]]
            self.input_output_tensors_shapes[output.name] = shape

        # Form node objects
        if not self.graph.input:
            raise ValueError("No input tensors found in the ONNX model.")

        current_shape = self.input_output_tensors_shapes.get(self.graph.input[0].name).copy()
        if current_shape is None:
            raise ValueError(f"Input tensor '{self.graph.input[0].name}' shape not found.")

        for node in self.graph.node:
            inputs = []
            outputs = []
            attributes = []

            # The constant node is created automatically when adding a scalar to a tensor (There are also other cases)
            if node.op_type == "Constant":
                # Extract data type, raw data, and dimensions
                scalar_data_type = node.attribute[0].t.data_type  # Data type as integer code
                scalar_raw_data = node.attribute[0].t.raw_data  # Raw binary data
                dims = node.attribute[0].t.dims  # Tensor dimensions
                if len(dims) != 0:
                    # Ensure all dimensions are integers
                    dims = [int(dim) for dim in dims]
                    num_elements = int(np.prod(dims))  # Cast to int
                    format_char = self._get_data_type(scalar_data_type)

                    # Calculate expected bytes
                    bytes_per_element = struct.calcsize(format_char)
                    expected_bytes = bytes_per_element * num_elements
                    if len(scalar_raw_data) != expected_bytes:
                        raise ValueError(f"Expected {expected_bytes} bytes for Constant node '{node.name}', but got {len(scalar_raw_data)} bytes.")

                    # Unpack all elements as a flat list
                    values_flat = struct.unpack(format_char * num_elements, scalar_raw_data)

                    # Reshape the flat list into nested lists based on dimensions
                    reshaped_values = np.reshape(values_flat, dims).tolist()

                    for out in node.output:
                        # Assign reshaped values to the output tensor
                        self.constant_values[out] = reshaped_values
                        outputs.append({'name': out, 'shape': list(dims)})
                        self.intermediate_tensors_shapes[out] = list(dims)

                        # Assign the 'value' attribute as the nested list of values
                        attributes.append({'name': 'value', 'value': reshaped_values})
                else:
                    for out in node.output:
                        outputs.append({'name': out, 'shape': 1})
                        self.intermediate_tensors_shapes[out] = 1       #add output tensors and their shapes to this dict for all nodes to use them in nodes like concat
                        self.constant_values[out] = struct.unpack(self._get_data_type(scalar_data_type), scalar_raw_data)[0]  #value of constant to be used in nodes like reshape
                    for attribute in node.attribute:
                        attributes.append({'name': attribute.name, 'value': struct.unpack(self._get_data_type(scalar_data_type), scalar_raw_data)[0]}) # converting bytes value based on data type

            elif node.op_type in ["Add", "Sub"]:
                inputs.append({'name': node.input[0], 'shape': current_shape.copy()})
                if node.input[1] in self.initializer_shapes:  # It's not a scalar, but a vector
                    inputs.append({'name': node.input[1], 'shape': current_shape.copy()})
                else:  # It's a scalar
                    inputs.append({'name': node.input[1], 'shape': [1]})  # Use list
                outputs.append({'name': node.output[0], 'shape': current_shape.copy()})
                self.intermediate_tensors_shapes[node.output[0]] = current_shape.copy()

            elif node.op_type == "MatMul":
                shape_weights = list(self.initializer_shapes.get(node.input[1], np.array(self.constant_values[node.input[1]]).shape))
                shape_input = current_shape.copy()
                shape_output = shape_input[:-1] + shape_weights[1:]  # Following matrix multiplication rules
                current_shape = shape_output.copy()

                inputs.append({'name': node.input[0], 'shape': shape_input})
                inputs.append({'name': node.input[1], 'shape': shape_weights})
                outputs.append({'name': node.output[0], 'shape': shape_output})
                self.intermediate_tensors_shapes[node.output[0]] = shape_output

            elif node.op_type == "Relu":
                inputs.append({'name': node.input[0], 'shape': current_shape.copy()})
                outputs.append({'name': node.output[0], 'shape': current_shape.copy()})
                self.intermediate_tensors_shapes[node.output[0]] = current_shape.copy()

            elif node.op_type == "Gemm":
                shape_weights = list(self.initializer_shapes.get(node.input[1], [1]))
                shape_bias = list(self.initializer_shapes.get(node.input[2], [1]))
                shape_input = [shape_weights[1]]
                shape_output = [shape_bias[0]]
                current_shape = shape_output.copy()

                inputs.append({'name': node.input[0], 'shape': shape_input})
                inputs.append({'name': node.input[1], 'shape': shape_weights})
                inputs.append({'name': node.input[2], 'shape': shape_bias})
                outputs.append({'name': node.output[0], 'shape': shape_output})
                self.intermediate_tensors_shapes[node.output[0]] = shape_output

                for attribute in node.attribute:
                    if attribute.type == onnx.AttributeProto.FLOAT:
                        value = attribute.f
                    elif attribute.type == onnx.AttributeProto.INT:
                        value = attribute.i
                    else:
                        value = None
                    attributes.append({'name': attribute.name, 'value': value})

            elif node.op_type == "Concat":
                # Assume axis=0
                axis = 0

                # Initialize output_shape based on the first input tensor
                first_input = node.input[0]
                first_input_shape = self.intermediate_tensors_shapes.get(first_input)

                output_shape = first_input_shape.copy()

                # Iterate over all input tensors to compute the output shape
                for input_name in node.input:
                    input_shape = self.intermediate_tensors_shapes.get(input_name)

                    # Sum the dimensions along the concatenation axis
                    output_shape[axis] += input_shape[axis]

                for input_name in node.input:
                    input_shape = self.intermediate_tensors_shapes.get(input_name)
                    inputs.append({'name': input_name, 'shape': input_shape})

                outputs.append({'name': node.output[0], 'shape': output_shape.copy()})
                self.intermediate_tensors_shapes[node.output[0]] = output_shape.copy()

            elif node.op_type == "Reshape":
                shape_tensor_input = current_shape.copy()
                new_shape = list(self.constant_values.get(node.input[1]))
                if new_shape == -1:
                    shape_tensor_out = [1]
                else:
                    shape_tensor_out = list(new_shape)  # Ensure list

                inputs.append({'name': node.input[0], 'shape': shape_tensor_input})
                inputs.append({'name': node.input[1], 'shape': [1]})  # Assuming scalar
                outputs.append({'name': node.output[0], 'shape': shape_tensor_out})
                self.intermediate_tensors_shapes[node.output[0]] = shape_tensor_out
                current_shape = shape_tensor_out.copy()

            elif node.op_type == "Flatten":
                # Retrieve the axis attribute
                axis_attribute = None
                for attribute in node.attribute:
                    if attribute.name == 'axis':
                        axis_attribute = attribute.i
                        break
                if axis_attribute is None:
                    raise ValueError(f"Flatten node '{node.name}' is missing the 'axis' attribute.")
                # Enforce axis=0
                if axis_attribute != 1:
                    raise ValueError(f"Axis attribute value of {node.name} node is not equal to 1.")
                # Compute the flattened dimension by collapsing all dimensions from axis=0 onward
                flattened_dim = 1
                for dim in current_shape:
                    flattened_dim *= dim
                # Define the output shape: [flattened_dim]
                shape_tensor_out = [flattened_dim]
                # Append input details
                inputs.append({'name': node.input[0], 'shape': current_shape})
                # Append output details
                outputs.append({'name': node.output[0], 'shape': shape_tensor_out})
                self.intermediate_tensors_shapes[node.output[0]] = shape_tensor_out.copy()
                # Record the axis attribute (optional, based on your library's needs)
                attributes.append({'name': 'axis', 'value': axis_attribute})


            elif node.op_type == "Shape":
                shape_tensor_input1 = current_shape.copy()
                shape_tensor_out = shape_tensor_input1.copy()
                inputs.append({'name': node.input[0], 'shape': shape_tensor_input1})
                # inputs.append({'name': node.input[1], 'shape': []})
                outputs.append({'name': node.output[0], 'shape': shape_tensor_out})
                self.intermediate_tensors_shapes[node.output[0]] = shape_tensor_out.copy()
                # attributes.append({'name': node.attribute[0].name, 'value': 0})

            elif node.op_type == "Gather":
                shape_tensor_input = current_shape.copy()
                inputs.append({'name': node.input[0], 'shape': shape_tensor_input})
                inputs.append({'name': node.input[1], 'shape': []})
                outputs.append({'name': node.output[0], 'shape': [1]})
                self.intermediate_tensors_shapes[node.output[0]] = [1]
                attributes.append({'name': node.attribute[0].name, 'value': 0})

            elif node.op_type == "Unsqueeze":
                axes_values = [int(attr.i) for attr in node.attribute if attr.name == "axes"]
                shape_tensor_input = current_shape.copy()
                shape_tensor_out = self._unsqueeze_shape(current_shape.copy(), axes_values)
                inputs.append({'name': node.input[0], 'shape': shape_tensor_input})
                outputs.append({'name': node.output[0], 'shape': self._unsqueeze_shape(current_shape, axes_values)})
                self.intermediate_tensors_shapes[node.output[0]] = self._unsqueeze_shape(current_shape, axes_values)
                attributes.append({'name': 'axes', 'value': axes_values})
                current_shape = shape_tensor_out.copy()

            elif node.op_type == "Conv":
                # Retrieve shapes
                shape_tensor_input = current_shape.copy()
                shape_weights = self.initializer_shapes.get(node.input[1])
                shape_bias = self.initializer_shapes.get(node.input[2]) if node.input[2] else None
                # shape_kernel =  [shape_weights[2], shape_weights[3]]

                # Extract attributes with defaults
                pads = [0, 0, 0, 0]  # [pad_top, pad_left, pad_bottom, pad_right]
                strides = [1, 1]
                dilations = [1, 1]
                group = 1

                for attr in node.attribute:
                    if attr.name == 'pads':
                        pads = list(attr.ints)
                    elif attr.name == 'strides':
                        strides = list(attr.ints)
                    elif attr.name == 'dilations':
                        dilations = list(attr.ints)
                    elif attr.name == 'group':
                        group = attr.i

                # Calculating output shape
                batch_size = 1       #batch size always 1
                channels, height_in, width_in = shape_tensor_input
                feature_maps, C_group, kernel_height, kernel_width = shape_weights
                pad_top, pad_left, pad_bottom, pad_right = pads
                stride_h, stride_w = strides
                dilation_h, dilation_w = dilations

                height_out = ((height_in + pad_top + pad_bottom - dilation_h * (kernel_height - 1) - 1) // stride_h) + 1
                width_out = ((width_in + pad_left + pad_right - dilation_w * (kernel_width - 1) - 1) // stride_w) + 1
                output_shape = [feature_maps, height_out, width_out]

                inputs.append({'name': node.input[0], 'shape': shape_tensor_input})
                inputs.append({'name': node.input[1], 'shape': list(shape_weights)})
                if node.input[2]:
                    inputs.append({'name': node.input[2], 'shape': list(shape_bias)})
                outputs.append({'name': node.output[0], 'shape': output_shape})

                self.intermediate_tensors_shapes[node.output[0]] = output_shape.copy()

                current_shape = output_shape.copy()

                attributes = {
                    'pads': pads,
                    'strides': strides,
                    'dilations': dilations,
                    'group': group
                }

            elif node.op_type == "MaxPool":
                shape_tensor_input = current_shape.copy()

                # Default values of attributes
                kernel_shape = [1, 1]
                strides = [1, 1]
                pads = [0, 0, 0, 0]
                ceil_mode = 0
                dilations = [1, 1]

                for attr in node.attribute:
                    if attr.name == 'kernel_shape':
                        kernel_shape = list(attr.ints)
                    elif attr.name == 'strides':
                        strides = list(attr.ints)
                    elif attr.name == 'pads':
                        pads = list(attr.ints)
                    elif attr.name == 'ceil_mode':
                        ceil_mode = attr.i
                    elif attr.name == 'dilations':
                        dilations = list(attr.ints)

                batch_size = 1 #default patch size
                channels, height_in, width_in = shape_tensor_input
                kernel_height, kernel_width = kernel_shape
                stride_h, stride_w = strides
                pad_top, pad_left, pad_bottom, pad_right = pads

                # Compute output dimensions
                if ceil_mode:
                    height_out = math.ceil(((height_in + pad_top + pad_bottom) - kernel_height) / stride_h) + 1
                    width_out = math.ceil(((width_in + pad_left + pad_right) - kernel_width) / stride_w) + 1
                else:
                    height_out = math.floor(((height_in + pad_top + pad_bottom) - kernel_height) / stride_h) + 1
                    width_out = math.floor(((width_in + pad_left + pad_right) - kernel_width) / stride_w) + 1

                shape_tensor_output = [channels, height_out, width_out]

                inputs.append({'name': node.input[0], 'shape': shape_tensor_input})
                outputs.append({'name': node.output[0], 'shape': shape_tensor_output})
                current_shape = output_shape.copy()
                # store attributes for use in operators
                attributes = {
                    'pads': pads,
                    'strides': strides,
                    'dilations': dilations,
                    'ceil_mode': ceil_mode,
                    'kernel_shape': kernel_shape
                }

            elif node.op_type == "AveragePool":
                shape_tensor_input = current_shape.copy()

                kernel_shape = [1, 1]
                strides = [1, 1]
                pads = [0, 0, 0, 0]
                ceil_mode = 0
                dilations = [1, 1]
                count_include_pad = 0

                for attr in node.attribute:
                    if attr.name == 'kernel_shape':
                        kernel_shape = list(attr.ints)
                    elif attr.name == 'strides':
                        strides = list(attr.ints)
                    elif attr.name == 'pads':
                        pads = list(attr.ints)
                    elif attr.name == 'ceil_mode':
                        ceil_mode = attr.i
                    elif attr.name == 'dilations':
                        dilations = list(attr.ints)
                    elif attr.name == 'count_include_pad':
                       count_include_pad = attr.i

                batch_size = 1 #default patch size
                channels, height_in, width_in = shape_tensor_input
                kernel_height, kernel_width = kernel_shape
                stride_h, stride_w = strides
                pad_top, pad_left, pad_bottom, pad_right = pads

                if ceil_mode:
                    height_out = math.ceil(((height_in + pad_top + pad_bottom) - kernel_height) / stride_h) + 1
                    width_out = math.ceil(((width_in + pad_left + pad_right) - kernel_width) / stride_w) + 1
                else:
                    height_out = math.floor(((height_in + pad_top + pad_bottom) - kernel_height) / stride_h) + 1
                    width_out = math.floor(((width_in + pad_left + pad_right) - kernel_width) / stride_w) + 1

                shape_tensor_output = [channels, height_out, width_out]

                inputs.append({'name': node.input[0], 'shape': shape_tensor_input})
                outputs.append({'name': node.output[0], 'shape': shape_tensor_output})

                attributes = {
                    'kernel_shape': kernel_shape,
                    'strides': strides,
                    'pads': pads,
                    'ceil_mode':  ceil_mode,
                    'dilations': dilations,
                    'count_include_pad': count_include_pad
                }

                self.nodes.append({
                    'name': node.name,
                    'type': node.op_type,
                    'input': inputs,
                    'output': outputs,
                    'attributes': attributes,
                    'initializers': self.initializer_values,
                    'constants': self.constant_values
                })

            elif node.op_type == "Dropout":
                shape_tensor_input = current_shape.copy()
                shape_tensor_output = current_shape.copy()
                shape_tensor_mask = current_shape.copy()

                inputs.append({'name': node.input[0], 'shape': shape_tensor_input})
                outputs.append({'name': node.output[0], 'shape': shape_tensor_output})

                # Check if a mask is present
                if len(node.output) > 1:
                    outputs.append({'name': node.output[1], 'shape': shape_tensor_mask})

                # Default values of attributes
                ratio = 0.5
                training_mode = False

                for attr in node.attribute:
                    if attr.name == 'ratio':
                        ratio = attr.f
                    elif attr.name == 'training_mode':
                        training_mode = attr.i

                attributes = {
                    'ratio': ratio,
                    'training_mode': training_mode
                }

                self.nodes.append({
                    'name': node.name,
                    'type': node.op_type,
                    'input': inputs,
                    'output': outputs,
                    'attributes': attributes,
                    'initializers': self.initializer_values,
                    'constants': self.constant_values
                })

            else:
                raise NotImplementedError(f"Operator {node.op_type} is not supported.")

            self.nodes.append({
                'name': node.name,
                'type': node.op_type,
                'input': inputs,
                'output': outputs,
                'attributes': attributes,
                'initializers': self.initializer_values,
                'constants': self.constant_values
            })

    # Translating data types in numbers form into letters to be used in struct.unpack()
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

    def _unsqueeze_shape(self, input_shape, axes):
        output_shape = input_shape.copy()
        for axis in sorted(axes):
            if axis < 0:
                axis += len(output_shape) + 1
            output_shape.insert(axis, 1)
        return output_shape