from base_parser import BaseParser

class UnsqueezeParser(BaseParser):
    def parse(self, node, parser):
        axes_values = [int(attr.i) for attr in node.attribute if attr.name == "axes"]
        shape_tensor_input = parser.current_shape.copy()
        output_shape = parser._unsqueeze_shape(parser.current_shape.copy(), axes_values)
        inputs = [{'name': node.input[0], 'shape': shape_tensor_input}]
        outputs = [{'name': node.output[0], 'shape': output_shape}]
        parser.intermediate_tensors_shapes[node.output[0]] = output_shape
        parser.current_shape = output_shape.copy()
        attributes = [{'name': 'axes', 'value': axes_values}]
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })

    def _unsqueeze_shape(self, input_shape, axes):
        output_shape = input_shape.copy()
        for axis in sorted(axes):
            if axis < 0:
                axis += len(output_shape) + 1
            output_shape.insert(axis, 1)
        return output_shape