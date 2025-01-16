from base_parser import BaseParser
import math

class MaxPoolParser(BaseParser):
    def parse(self, node, parser):
        shape_tensor_input = parser.current_shape.copy()
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
        inputs = [{'name': node.input[0], 'shape': shape_tensor_input}]
        outputs = [{'name': node.output[0], 'shape': shape_tensor_output}]
        parser.current_shape = shape_tensor_output.copy()

        attributes = {
            'kernel_shape': kernel_shape,
            'strides': strides,
            'pads': pads,
            'ceil_mode': ceil_mode,
            'dilations': dilations
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
