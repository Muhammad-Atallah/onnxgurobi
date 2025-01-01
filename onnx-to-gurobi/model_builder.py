from itertools import product
from gurobipy import Model, GRB
from parser import ONNXParser
from operators.operator_factory import OperatorFactory

class ONNXToGurobi:
    def __init__(self, onnx_model_path):
        self.parser = ONNXParser(onnx_model_path)
        self.model = Model("NeuralNetwork")
        self.variables = {}
        self.initializers = self.parser.initializer_values
        self.nodes = self.parser.nodes
        self.operator_factory = OperatorFactory()

    def create_variables(self):
        # Create variables for inputs and outputs
        for tensor_name, shape in self.parser.input_output_tensors_shapes.items():
            indices = self._generate_indices(shape)
            self.variables[tensor_name] = self.model.addVars(
                indices,
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                name=tensor_name
            )

        # Create variables for intermediate nodes
        for node in self.nodes:
            output_name = node['output'][0]['name']
            if node['type'] == "Constant":
                # Assign constant value directly without creating a variable
                self.variables[output_name] = node['attributes'][0]['value']
            elif node['type'] == "Relu":
                # For Relu, adding binary variables for activation
                shape = node['output'][0]['shape']
                indices = self._generate_indices(shape)
                var_input = self.variables[node["input"][0]["name"]]
                self.variables[f"relu_binary_{output_name}"] = self.model.addVars(
                    var_input.keys(),
                    vtype=GRB.BINARY,
                    name=f"relu_binary_{output_name}"
                )

                self.variables[output_name] = self.model.addVars(
                    indices,
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )
            else:
                shape = node['output'][0]['shape']
                indices = self._generate_indices(shape)
                self.variables[output_name] = self.model.addVars(
                    indices,
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )

    def build_model(self):
        self.create_variables()
        for node in self.nodes:
            if node['type'] != "Constant":
                operator = self.operator_factory.create_operator(node, self.initializers)
                operator.apply_constraints(self.model, self.variables)

    def get_gurobi_model(self):
        return self.model

    def _generate_indices(self, shape):
        if isinstance(shape, int):  # Single integer case
            return range(shape)
        elif isinstance(shape, (list, tuple)):  # Multi-dimensional case
            return product(*[range(dim) for dim in shape])
        else:
            raise ValueError("Shape must be an integer or a list/tuple of integers.")
