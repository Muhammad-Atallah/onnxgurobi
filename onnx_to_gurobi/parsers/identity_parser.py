from .base_parser import BaseParser
import numpy as np

class IdentityParser(BaseParser):
    """
    Parses the ONNX Identity node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Identity operation.

    """
    def parse(self, node, parser):
        """
        Parses the Identity node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the Identity node.
        """
        input_values = parser.initializer_values[node.input[0]]
        input_shape = list(np.array(input_values).shape)
        output_shape = input_shape

        inputs = [{'name': node.input[0], 'shape': input_shape}]
        outputs = [{'name': node.output[0], 'shape': output_shape}]
        attributes = [{'name': 'Identity', 'value': input_values}]
        parser.intermediate_tensors_shapes[node.output[0]] = output_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })