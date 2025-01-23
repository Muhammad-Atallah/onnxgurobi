from .base_parser import BaseParser

class ConcatParser(BaseParser):
    """
    Parses the ONNX Concat node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Concat operation.

    """
    def parse(self, node, parser):
        """
        Parses the Concat node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Updates `parser.current_shape` with the shape of the output.
            - Appends a new entry to `parser.nodes` describing the Concat node.
        """

        axis = 0
        first_input = node.input[0]
        first_input_shape = parser.intermediate_tensors_shapes.get(first_input)
        output_shape = first_input_shape.copy()
        inputs = []

        for input_name in node.input:
            input_shape = parser.intermediate_tensors_shapes.get(input_name)
            output_shape[axis] += input_shape[axis]
            inputs.append({'name': input_name, 'shape': input_shape})

        outputs = [{'name': node.output[0], 'shape': output_shape.copy()}]
        parser.intermediate_tensors_shapes[node.output[0]] = output_shape.copy()
        parser.current_shape = output_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': [{'name': 'axis', 'value': axis}],
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
