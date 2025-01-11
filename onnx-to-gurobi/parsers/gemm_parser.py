from base_parser import BaseParser
import onnx

class GemmParser(BaseParser):
    def parse(self, node, parser):
        current_shape = parser.current_shape.copy()
        shape_weights = list(parser.initializer_shapes.get(node.input[1], [1]))
        shape_bias = list(parser.initializer_shapes.get(node.input[2], [1]))
        shape_input = [shape_weights[1]]
        shape_output = [shape_bias[0]]
        parser.current_shape = shape_output.copy()

        inputs = [
            {'name': node.input[0], 'shape': shape_input},
            {'name': node.input[1], 'shape': shape_weights},
            {'name': node.input[2], 'shape': shape_bias}
        ]
        outputs = [{'name': node.output[0], 'shape': shape_output}]
        attributes = []
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.FLOAT:
                value = attribute.f
            elif attribute.type == onnx.AttributeProto.INT:
                value = attribute.i
            else:
                value = None
            attributes.append({'name': attribute.name, 'value': value})

        parser.intermediate_tensors_shapes[node.output[0]] = shape_output
        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
