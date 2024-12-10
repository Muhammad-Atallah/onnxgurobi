from itertools import product
from gurobipy import Model, GRB
from parser import ONNXParser
from gemm import GemmOperator
from add import AddOperator
from matmul import MatMul
from relu import ReLUOperator
from sub import SubOperator
from concat import ConcatOperator
from reshape import ReshapeOperator
from flatten import FlattenOperator
from gather import GatherOperator
from unsqueeze import UnsqueezeOperator
from conv import ConvOperator
from maxpool import MaxPoolOperator
from averagepool import AveragePoolOperator
from dropout import DropoutOperator

class ONNXToGurobi:
    def __init__(self, onnx_model_path):
        self.parser = ONNXParser(onnx_model_path)
        self.model = Model("NeuralNetwork")
        self.variables = {}
        self.initializers = self.parser.initializer_values
        self.nodes = self.parser.nodes
        self.node_handlers = {
            'Gemm'          : GemmOperator,
            'Add'           : AddOperator,
            'MatMul'        : MatMul,
            'Relu'          : ReLUOperator,
            'Sub'           : SubOperator,
            'Concat'        : ConcatOperator,
            'Reshape'       : ReshapeOperator,
            'Flatten'       : FlattenOperator,
            'Unsqueeze'     : UnsqueezeOperator,
            'Gather'        : GatherOperator,
            'Conv'          : ConvOperator,
            'MaxPool'       : MaxPoolOperator,
            'AveragePool'   : AveragePoolOperator,
            'Dropout'       : DropoutOperator
        }

    def create_variables(self):
        # Creating variables for inputs and outputs
        for tensor_name, shape in self.parser.input_output_tensors_shapes.items():
            indices = self._generate_indices(shape)
            self.variables[tensor_name] = self.model.addVars(
                indices,
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                name=tensor_name
            )

        # Creating variables for intermediate nodes
        for node in self.nodes:
            output_name = node['output'][0]['name']
            if node['type'] == "Constant":
                self.variables[output_name] = node['attributes'][0]['value']        #No need to create a variable for the scalar. Just saving the value of the constant here.
            elif node['type'] == "Relu":            #for Relu, in addition to adding variables to the output tensor, a binary variable has to be added for each element in the input variable
                shape = node['output'][0]['shape']
                indices = self._generate_indices(shape)
                var_input = self.variables[node["input"][0]["name"]]
                self.variables[f"relu_binary_{output_name}"] = self.model.addVars(
                    var_input.keys(),
                    vtype=GRB.BINARY,
                    name=f"relu_binary_{output_name}")

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
            if(node['type'] != "Constant"):
                op_type = node['type']
                handler_class = self.node_handlers.get(op_type)
                if handler_class is None:
                    raise NotImplementedError(f"Operator {op_type} is not supported.")
                if op_type in ['Add', 'Sub', 'Concat', 'Reshape', 'Flatten', 'Relu', 'Unsqueeze', 'MaxPool', 'AveragePool', 'Dropout']:
                    handler = handler_class(node)
                else:
                    handler = handler_class(node, self.initializers)
                handler.apply_constraints(self.model, self.variables)

    def get_gurobi_model(self):
        return self.model

    def _generate_indices(self, shape):
        if isinstance(shape, int):  # Single integer case
            return range(shape)
        elif isinstance(shape, (list, tuple)):  # Multi dimensional case
            return product(*[range(dim) for dim in shape])
        else:
            raise ValueError("Shape must be an integer or a list/tuple of integers.")

