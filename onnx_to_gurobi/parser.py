import onnx
import numpy as np
from .parsers.parser_factory import ParserFactory
from .internalOnnx import InternalONNXRepresentation

class ONNXParser:
    """
    Parses an ONNX model, extracting initializers, shapes for inputs and outputs,
    and creating an internal representation of nodes for processing.

    Attributes:
        onnx_model (onnx.ModelProto): The loaded ONNX model object.
        graph (onnx.GraphProto): The main graph of the ONNX model.
        initializer_shapes (dict): Maps each initializer name to its shape (list of ints).
        initializer_values (dict): Maps each initializer name to its actual values (NumPy array).
        input_output_tensors_shapes (dict): Stores shapes for all input and output tensors.
        intermediate_tensors_shapes (dict): Tracks shapes for intermediate nodes generated by the parser.
        constant_values (dict): Stores constant values.
        nodes (list): The internal representation of nodes, each a dict describing
            inputs, outputs, attributes, and references to initializers.
        current_shape (list): Keeps track of the shape of the most recent node's output.
        node_parser_factory (ParserFactory): Responsible for providing parser classes for each node type.
    """
    def __init__(self, onnx_model_path):
        """
        Initializes the ONNXParser by loading the model and preparing data structures.

        Args:
            onnx_model_path (str): Path to the ONNX file to be parsed.

        """
        self.onnx_model = onnx.load(onnx_model_path)
        self.graph = self.onnx_model.graph
        self.initializer_shapes = {}
        self.initializer_values = {}
        self.input_output_tensors_shapes = {}
        self.intermediate_tensors_shapes = {}
        self.constant_values = {}
        self.nodes = []
        self.current_shape = None
        self.node_parser_factory = ParserFactory()
        self._parse_model()

    def _parse_model(self):
        """
        Parses the ONNX graph to populate initializers, input/output shapes,
        and create internal node representations.

        Iterates through each node in the ONNX graph, retrieving a parser
        from `node_parser_factory` according to the node type, and invokes
        the parser's `parse` method to update the `nodes` dictionary.

        Raises:
            ValueError: If the ONNX model does not define any input tensors.
        """
        for initializer in self.graph.initializer:
            initializer_array = onnx.numpy_helper.to_array(initializer)
            self.initializer_shapes[initializer.name] = list(initializer_array.shape)
            self.initializer_values[initializer.name] = initializer_array

        for input in self.graph.input:
            if input.name in self.initializer_shapes:
                continue
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim[1:]]
            if len(shape) == 1:
                shape = [shape[0]]
            self.input_output_tensors_shapes[input.name] = shape

        for output in self.graph.output:
            shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim[1:]]
            if len(shape) == 1:
                shape = [shape[0]]
            self.input_output_tensors_shapes[output.name] = shape

        if not self.graph.input:
            raise ValueError("No input tensors to the ONNX model found.")

        self.current_shape = self.input_output_tensors_shapes[self.graph.input[0].name].copy()

        for node in self.graph.node:
            parser = self.node_parser_factory.get_parser(node.op_type)
            parser.parse(self, node, self)

        return InternalONNXRepresentation(self)

