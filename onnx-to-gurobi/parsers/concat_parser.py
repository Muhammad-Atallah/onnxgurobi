from base_parser import BaseParser

class ConcatParser(BaseParser):
    def parse(self, node, parser):
        axis = 0
        first_input = node.input[0]
        first_input_shape = parser.intermediate_tensors_shapes.get(first_input)
        output_shape = first_input_shape.copy()
        inputs = []
        outputs = []

        for input_name in node.input:
            input_shape = parser.intermediate_tensors_shapes.get(input_name)
            output_shape[axis] += input_shape[axis]
            inputs.append({'name': input_name, 'shape': input_shape})

        outputs.append({'name': node.output[0], 'shape': output_shape.copy()})
        parser.intermediate_tensors_shapes[node.output[0]] = output_shape.copy()
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': [{'name': 'axis', 'value': axis}],
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
