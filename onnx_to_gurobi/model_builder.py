from itertools import product
from gurobipy import Model, GRB
from .operators.operator_factory import OperatorFactory
from parser import ONNXParser
from utils import _generate_indices

class ONNXToGurobi:
    """
    Converts an ONNX model to a Gurobi model by parsing nodes and creating
    corresponding constraints for each operator.

    Attributes:
        parser (ONNXParser): The parser responsible for reading the ONNX model and extracting nodes, initializers, and shapes.
        model (gurobipy.Model): The Gurobi model being constructed.
        variables (dict): A mapping of tensor names to either Gurobi variables or constant values.
        initializers (dict): Contains the initial values from the parsed ONNX model.
        nodes (list): A list of dictionaries, each representing an ONNX node extracted by the parser.
        operator_factory (OperatorFactory): Responsible for creating operator instances based on node types.
    """
    def __init__(self, onnx_model_path):
        """
        Initializes the ONNXToGurobi class with a path to an ONNX model.

        Args:
            onnx_model_path (str): The path to the ONNX model file to be parsed.
        """
        self.parser = ONNXParser(onnx_model_path)
        self.model = Model("NeuralNetwork")
        self.variables = {}
        self.initializers = self.parser.initializer_values
        self.nodes = self.parser.nodes
        self.operator_factory = OperatorFactory()

    def create_variables(self):
        """
        Creates Gurobi variables for the input/output tensors and intermediate nodes.

        """
        # Create variables for inputs and outputs
        for tensor_name, shape in self.parser.input_output_tensors_shapes.items():
            indices = _generate_indices(shape)
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
                # Constants are not model variables
                if 'attributes' in node and node['attributes']:
                    self.variables[output_name] = node['attributes'][0]['value']
                else:
                    self.variables[output_name] = 0

            elif node['type'] == "Identity":
                self.variables[output_name] = node['attributes'][0]['value']

            elif node['type'] == "Relu":
                shape = node['output'][0]['shape']
                indices = _generate_indices(shape)
                var_input = self.variables[node["input"][0]["name"]]

                # Create binary variables for ReLU indicator
                self.variables[f"relu_binary_{output_name}"] = self.model.addVars(
                    var_input.keys(),
                    vtype=GRB.BINARY,
                    name=f"relu_binary_{output_name}"
                )

                # Create output variables
                self.variables[output_name] = self.model.addVars(
                    indices,
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )

            else:
                shape = node['output'][0]['shape']
                indices = _generate_indices(shape)
                self.variables[output_name] = self.model.addVars(
                    indices,
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )

    def build_model(self):
        """
        Constructs the Gurobi model by creating variables and applying operator constraints.

        """
        self.create_variables()
        for node in self.nodes:
            if node['type'] != "Constant":
                op_type = node['type']
                operator = self.operator_factory.create_operator(node, self.initializers)
                operator.apply_constraints(self.model, self.variables)

    def get_gurobi_model(self):
        """
        Retrieves the Gurobi model after all constraints have been added.

        Returns:
            gurobipy.Model: The constructed Gurobi model reflecting the ONNX graph.
        """
        return self.model
