from base_parser import BaseParser
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
    def parse(self, node, parser):
        input_shape = parser.initializer_shapes[node.input[0]]
        
        inputs = [{'name': node.input[0], 'shape': input_shape}]
        outputs = [{'name': node.output[0], 'shape': input_shape}]
        attributes = {}
        parser.intermediate_tensors_shapes[node.output[0]] = input_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })