from base_parser import BaseParser

class ReshapeParser(BaseParser):
    def parse(self, node, parser):
        shape_tensor_input = parser.current_shape.copy()
        new_shape = list(parser.constant_values.get(node.input[1]))
        shape_tensor_out = list(new_shape) if new_shape != -1 else [1]
        filtered_shape_tensor_out = [dim for dim in shape_tensor_out if dim > 0]
        inputs = [
            {'name': node.input[0], 'shape': shape_tensor_input},
            {'name': node.input[1], 'shape': new_shape}
        ]
        outputs = [{'name': node.output[0], 'shape': filtered_shape_tensor_out}]
        parser.intermediate_tensors_shapes[node.output[0]] = filtered_shape_tensor_out
        parser.current_shape = filtered_shape_tensor_out.copy()
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': [],
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
