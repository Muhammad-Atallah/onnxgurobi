from base_parser import BaseParser
import math

class ConvParser(BaseParser):
    def parse(self, node, parser):
        shape_tensor_input = parser.current_shape.copy()
        shape_weights = parser.initializer_shapes.get(node.input[1])
        shape_bias = parser.initializer_shapes.get(node.input[2]) if node.input[2] else None

        pads = [0, 0, 0, 0]
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

        channels, height_in, width_in = shape_tensor_input
        feature_maps, C_group, kernel_height, kernel_width = shape_weights
        pad_top, pad_left, pad_bottom, pad_right = pads
        stride_h, stride_w = strides
        dilation_h, dilation_w = dilations

        height_out = ((height_in + pad_top + pad_bottom - dilation_h * (kernel_height - 1) - 1) // stride_h) + 1
        width_out = ((width_in + pad_left + pad_right - dilation_w * (kernel_width - 1) - 1) // stride_w) + 1
        output_shape = [feature_maps, height_out, width_out]

        inputs = [{'name': node.input[0], 'shape': shape_tensor_input},
                  {'name': node.input[1], 'shape': list(shape_weights)}]
        if node.input[2]:
            inputs.append({'name': node.input[2], 'shape': list(shape_bias)})
        outputs = [{'name': node.output[0], 'shape': output_shape}]
        parser.intermediate_tensors_shapes[node.output[0]] = output_shape.copy()
        parser.current_shape = output_shape.copy()

        attributes = {
            'pads': pads,
            'strides': strides,
            'dilations': dilations,
            'group': group
        }
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
