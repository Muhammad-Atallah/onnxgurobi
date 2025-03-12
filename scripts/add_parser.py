from base_parser import BaseParser

class AddParser(BaseParser):
    """
    Parses the ONNX Add node.

    This parser extracts the necessary inputs and outputs, determines their
    shapes, and adds an entry to the parser's node list representing the
    Add operation.

    """

    def parse(self, node, parser):
        """
        Parses the Add node and updates the parser's internal representation.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            parser: The main parser module, which maintains information like
                current_shape, intermediate_tensors_shapes, and the node list.

        Returns:
            None: The method updates the parser in place.

        Side Effects:
            - Updates `parser.intermediate_tensors_shapes` with the output of the node and its shape.
            - Appends a new entry to `parser.nodes` describing the Add node.
        """
        current_shape = parser.current_shape.copy()
        inputs = [{'name': node.input[0], 'shape': current_shape.copy()}]

        # Second input is either in the initializers, the parser.intermediate_tensors_shapes, or it's a constant of shape [1]
        if node.input[1] in parser.initializer_shapes:
            inputs.append({'name': node.input[1], 'shape': current_shape.copy()})
        elif node.input[1] in parser.intermediate_tensors_shapes:
            inputs.append({'name': node.input[1], 'shape': parser.intermediate_tensors_shapes[node.input[1]]})
        else:
            inputs.append({'name': node.input[1], 'shape': [1]})

        outputs = [{'name': node.output[0], 'shape': current_shape.copy()}]
        parser.intermediate_tensors_shapes[node.output[0]] = current_shape.copy()

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': {},
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
