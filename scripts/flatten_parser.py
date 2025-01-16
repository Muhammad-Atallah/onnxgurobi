from base_parser import BaseParser

class FlattenParser(BaseParser):
    def parse(self, node, parser):
        axis_attribute = None
        for attribute in node.attribute:
            if attribute.name == 'axis':
                axis_attribute = attribute.i
                break
        if axis_attribute is None or axis_attribute != 1:
            raise ValueError(f"Unsupported axis in Flatten node '{node.name}'.")
        current_shape = parser.current_shape.copy()
        flattened_dim = 1
        for dim in current_shape:
            flattened_dim *= dim
        shape_tensor_out = [flattened_dim]
        inputs = [{'name': node.input[0], 'shape': current_shape}]
        outputs = [{'name': node.output[0], 'shape': shape_tensor_out}]
        parser.intermediate_tensors_shapes[node.output[0]] = shape_tensor_out.copy()
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': [{'name': 'axis', 'value': axis_attribute}],
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
