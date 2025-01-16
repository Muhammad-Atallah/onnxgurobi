from base_parser import BaseParser

class ShapeParser(BaseParser):
    def parse(self, node, parser):
        shape_tensor_input = parser.current_shape.copy()
        shape_tensor_out = shape_tensor_input.copy()

        inputs = [{'name': node.input[0], 'shape': shape_tensor_input}]
        outputs = [{'name': node.output[0], 'shape': shape_tensor_out}]

        parser.current_shape = shape_tensor_out.copy()
        parser.intermediate_tensors_shapes[node.output[0]] = shape_tensor_out.copy()

        attributes = [{'name': 'axis', 'value': 0}]

        return {
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        }
