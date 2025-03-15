from .base_parser import BaseParser
import numpy as np

class BatchNormalization(BaseParser):
    """
    Parses the ONNX BatchNormalization node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    BatchNormalization operation.

    """
    def parse(self, node, parser):
        """
        Parses the BatchNormalization node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the BatchNormalization node.
        """
        current_shape = parser.current_shape.copy()
        inputs = [
            {'name': node.input[0], 'shape': current_shape},
            {'name': node.input[1], 'shape': current_shape},
            {'name': node.input[2], 'shape': current_shape},
            {'name': node.input[3], 'shape': current_shape},
            {'name': node.input[4], 'shape': current_shape}
        ]
        outputs = [{'name': node.output[0], 'shape': current_shape}]
        attributes = {}

        for attribute in node.attribute:
            attributes[attribute.name] = attribute.f

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