from .base_parser import BaseParser

class DropoutParser(BaseParser):
    """
    Parses the ONNX Dropout node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Dropout operation.

    """
    def parse(self, node, parser):
        """
        Parses the Dropout node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the Dropout node.
        """

        inputs = [{'name': node.input[0], 'shape': parser.current_shape.copy()}]
        outputs = [{'name': node.output[0], 'shape': parser.current_shape.copy()}]
        if len(node.output) > 1:
            outputs.append({'name': node.output[1], 'shape': parser.current_shape.copy()})
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
        parser.intermediate_tensors_shapes[node.output[0]] = parser.current_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
