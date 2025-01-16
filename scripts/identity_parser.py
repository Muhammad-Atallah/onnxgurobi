from base_parser import BaseParser
import numpy as np

class IdentityParser(BaseParser):
    def parse(self, node, parser):
        inputs = []
        outputs = []
        attributes = []

        input_values = parser.initializer_values[node.input[0]]
        input_shape = list(np.array(input_values).shape)
        output_shape = input_shape

        inputs.append({'name': node.input[0], 'shape': input_shape})
        outputs.append({'name': node.output[0], 'shape': output_shape})
        attributes.append({'value': input_values})


        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })