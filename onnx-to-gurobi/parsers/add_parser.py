from base_parser import BaseParser

class AddParser(BaseParser):
    def parse(self, node, parser):
        current_shape = parser.current_shape.copy()
        inputs = []
        outputs = []

        inputs.append({'name': node.input[0], 'shape': current_shape.copy()})
        if node.input[1] in parser.initializer_shapes:
            inputs.append({'name': node.input[1], 'shape': current_shape.copy()})
        else:
            inputs.append({'name': node.input[1], 'shape': [1]})
        outputs.append({'name': node.output[0], 'shape': current_shape.copy()})
        parser.intermediate_tensors_shapes[node.output[0]] = current_shape.copy()
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': [],
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
