import onnx

class ONNXParser:

    def __init__(self, model_path):
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.nodes = self.graph.node
        self.initializers = self._load_initializers(self.graph)
        self.inputs = self._load_inputs()
        self.outputs = self._load_outputs()

    def _load_initializers(self, graph):
        initializers = {}
        for initializer in graph.initializer:
            initializers[initializer.name] = onnx.numpy_helper.to_array(initializer)
        return initializers

    def _load_inputs(self):
        input_tensors = []
        for input_tensor in self.graph.input:
            input_tensors.append(self._parse_tensor(input_tensor))
        return input_tensors

    def _load_outputs(self):
        output_tensors = []
        for output_tensor in self.graph.output:
            output_tensors.append(self._parse_tensor(output_tensor))
        return output_tensors

    def _parse_tensor(self, tensor):
        return {'tensor_name': tensor.name, 'tensor_shape': [dim.dim_value for dim in tensor.type.tensor_type.shape.dim]}

    def get_graph(self):
        return self.graph

    def get_nodes(self):
        return self.nodes

    def get_initializers(self):
        return self.initializers

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs