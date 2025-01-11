from base_parser import BaseParser
import numpy as np

class BatchNormalization(BaseParser):
    def parse(self, node, parser):
        current_shape = parser.current_shape.copy()

        inputs = [
            {'name': node.input[0], 'shape': current_shape},
            {'name': node.input[1], 'shape': current_shape},
            {'name': node.input[2], 'shape': current_shape},
            {'name': node.input[3], 'shape': current_shape},
            {'name': node.input[4], 'shape': current_shape}
        ]
        outputs = [{'name': node.output[0], 'shape': current_shape}]
        attributes = [
            {'name': attributes[0].name, 'value': attributes[0].f},
            {'name': attributes[1].name, 'value': attributes[1].f}
        ]

        parser.intermediate_tensors_shapes[node.output[0]] = current_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })