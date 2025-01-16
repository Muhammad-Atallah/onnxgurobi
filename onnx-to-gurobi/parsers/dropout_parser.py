from base_parser import BaseParser

class DropoutParser(BaseParser):
    def parse(self, node, parser):
        shape_tensor_input = parser.current_shape.copy()
        shape_tensor_output = shape_tensor_input.copy()
        inputs = [{'name': node.input[0], 'shape': shape_tensor_input}]
        outputs = [{'name': node.output[0], 'shape': shape_tensor_output}]
        if len(node.output) > 1:
            outputs.append({'name': node.output[1], 'shape': shape_tensor_output})
        ratio = 0.5
        training_mode = False
        for attr in node.attribute:
            if attr.name == 'ratio':
                ratio = attr.f
            elif attr.name == 'training_mode':
                training_mode = attr.i
        attributes = [
            {"name": "ratio", "value": ratio},
            {"name": "training_mode", "value": training_mode}
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
