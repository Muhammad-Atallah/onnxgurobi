from base_parser import BaseParser

class ReluParser(BaseParser):
    def parse(self, node, parser):
        current_shape = parser.current_shape.copy()
        inputs = [{'name': node.input[0], 'shape': current_shape.copy()}]
        outputs = [{'name': node.output[0], 'shape': current_shape.copy()}]
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
