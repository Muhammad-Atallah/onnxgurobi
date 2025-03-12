from base_parser import BaseParser

class ReshapeParser(BaseParser):
    """
    Parses the ONNX Reshape node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Reshape operation.

    """
    def parse(self, node, parser):
        """
        Parses the Reshape node and updates the parser's internal representation.

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
            - Appends a new entry to `parser.nodes` describing the Reshape node.
        """
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
