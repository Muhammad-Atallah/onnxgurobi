from base_parser import BaseParser

class MatMulParser(BaseParser):
    def parse(self, node, parser):
        shape_weights = list(parser.initializer_shapes.get(node.input[1], [1]))
        shape_input = parser.current_shape.copy()
        shape_output = shape_input[:-1] + shape_weights[1:]
        parser.current_shape = shape_output.copy()

        inputs = [
            {'name': node.input[0], 'shape': shape_input},
            {'name': node.input[1], 'shape': shape_weights}
        ]
        outputs = [{'name': node.output[0], 'shape': shape_output}]
        parser.intermediate_tensors_shapes[node.output[0]] = shape_output
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': [],
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
