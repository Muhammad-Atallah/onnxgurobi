from base_parser import BaseParser

class ShapeParser(BaseParser):
    """
    Parses the ONNX Shape node.

    This parser extracts the necessary inputs, outputs and attributes, determines their
    shapes and values, and adds an entry to the parser's node list representing the
    Shape operation.

    """
    def parse(self, node, parser):
        """
        Parses the Shape node and updates the parser's internal representation.

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
            - Appends a new entry to `parser.nodes` describing the Shape node.
        """

        shape_tensor_input = parser.current_shape.copy()
        shape_tensor_out = shape_tensor_input.copy()

        inputs = [{'name': node.input[0], 'shape': shape_tensor_input}]
        outputs = [{'name': node.output[0], 'shape': shape_tensor_out}]

        parser.current_shape = shape_tensor_out.copy()
        parser.intermediate_tensors_shapes[node.output[0]] = shape_tensor_out.copy()

        attributes = {'axis' : 0}

        return {
            'name': node.name,
            'type': node.op_type,
            'input': inputs,
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        }
