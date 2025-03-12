from base_parser import BaseParser

class SubParser(BaseParser):
    """
    Parses the ONNX Sub node.

    This parser extracts the necessary inputs and outputs, determines their
    shapes, and adds an entry to the parser's node list representing the
    Sub operation.

    """
    def parse(self, node, parser):
        """
        Parses the Sub node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the Sub node.
        """
        current_shape = parser.current_shape.copy()
        inputs = []
        outputs = []
        if node.op_type == "Sub":
            inputs.append({'name': node.input[0], 'shape': current_shape.copy()})
            if node.input[1] in parser.initializer_shapes:
                inputs.append({'name': node.input[1], 'shape': current_shape.copy()})
            else:
                inputs.append({'name': node.input[1], 'shape': [1]})
            outputs.append({'name': node.output[0], 'shape': current_shape.copy()})
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
