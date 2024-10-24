from gurobipy import Model, GRB
from parser import ONNXParser
from gemm import GemmOperator

class ONNXToGurobi:
    def __init__(self, onnx_model_path):
        self.parser = ONNXParser(onnx_model_path)
        self.model = Model("NeuralNetwork")
        self.variables = {}
        self.initializers = self.parser.initializer_values
        self.nodes = self.parser.nodes
        self.node_handlers = {
            'Gemm': GemmOperator,
        }

    def create_variables(self):
        # Create variables for inputs and outputs
        for tensor_name, shape in self.parser.input_output_tensors_shapes.items():
            self.variables[tensor_name] = self.model.addVars(
                range(shape),
                vtype=GRB.CONTINUOUS,
                name=tensor_name
            )

        # Create variables for intermediate nodes
        for node in self.nodes:
            output_name = node['output'][0]['name']
            size = node['output'][0]['shape']
            self.variables[output_name] = self.model.addVars(
                range(size),
                vtype=GRB.CONTINUOUS,
                name=output_name
            )

    def build_model(self):
        self.create_variables()
        for node in self.nodes:
            op_type = node['type']
            handler_class = self.node_handlers.get(op_type)
            if handler_class is None:
                raise NotImplementedError(f"Operator {op_type} is not supported.")
            if op_type in ['Add', 'Sub']:
                handler = handler_class(node, self.nodes)
            else:
                handler = handler_class(node, self.initializers)
            handler.apply_constraints(self.model, self.variables)

    def get_gurobi_model(self):
        return self.model
