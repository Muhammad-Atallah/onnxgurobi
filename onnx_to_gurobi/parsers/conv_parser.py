from base_parser import BaseParser
import math

class ConvParser(BaseParser):
    """
    Parses the ONNX Conv node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Conv operation.

    """
    def parse(self, node, parser):
        """
        Parses the Conv node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Updates `parser.current_shape` with the shape of the output.
            - Appends a new entry to `parser.nodes` describing the Conv node.
        """
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

        attributes = [
            {"name": "pads", "value": pads},
            {"name": "strides", "value": strides},
            {"name": "dilations", "value": dilations},
            {"name": "group", "value": group}
        ]

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
