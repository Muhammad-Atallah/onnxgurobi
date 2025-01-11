from base_parser import BaseParser

class GatherParser(BaseParser):
    def parse(self, node, parser):
        current_shape = parser.current_shape.copy()
        inputs = [{'name': node.input[0], 'shape': current_shape.copy()}]
        outputs = [{'name': node.output[0], 'shape': [1]}]
        parser.intermediate_tensors_shapes[node.output[0]] = [1]
        attributes = []
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
