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

class ONNXToGurobi:
    def __init__(self, onnx_model_path):
        self.parser = ONNXParser(onnx_model_path)
        self.model = Model("NeuralNetwork")
        self.variables = {}
        self.initializers = self.parser.initializer_values
        self.nodes = self.parser.nodes
        self.node_handlers = {
            'Gemm'      : GemmOperator,
            'Add'       : AddOperator,
            'MatMul'    : MatMul,
            'Relu'      : ReLUOperator,
            'Sub'       : SubOperator,
            'Concat'    : ConcatOperator,
            'Reshape'   : ReshapeOperator,
            'Flatten'   : FlattenOperator
        }

    def create_variables(self):
        # Create variables for inputs and outputs
        for tensor_name, shape in self.parser.input_output_tensors_shapes.items():
            self.variables[tensor_name] = self.model.addVars(
                range(shape),
                vtype=GRB.CONTINUOUS,
                lb=-GRB.INFINITY,
                name=tensor_name
            )

        # Create variables for intermediate nodes
        for node in self.nodes:
            output_name = node['output'][0]['name']
            if node['type'] == "Constant":
                self.variables[output_name] = node['attributes'][0]['value']        #No need to create a variable for the scalar. Just saving the value of the constant here.
            elif node['type'] == "Relu":            #for Relu, in addition to adding variables to the output tensor, a binary variable has to be added for each element in the input variable
                shape = node['output'][0]['shape']
                var_input = self.variables[node["input"][0]["name"]]
                self.variables[f"relu_binary_{output_name}"] = self.model.addVars(
                    var_input.keys(),
                    vtype=GRB.BINARY,
                    name=f"relu_binary_{output_name}")

                self.variables[output_name] = self.model.addVars(
                    range(shape),
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )
            else:
                shape = node['output'][0]['shape']
                self.variables[output_name] = self.model.addVars(
                    range(shape),
                    vtype=GRB.CONTINUOUS,
                    lb=-GRB.INFINITY,
                    name=output_name
                )
    def build_model(self):
        self.create_variables()
        for node in self.nodes:
            if(node['type'] != "Constant"):
                op_type = node['type']
                print("op_type:", op_type)
                handler_class = self.node_handlers.get(op_type)
                if handler_class is None:
                    raise NotImplementedError(f"Operator {op_type} is not supported.")
                if op_type in ['Add', 'Sub', 'Concat', 'Reshape', 'Flatten']:
                    handler = handler_class(node)
                else:
                    handler = handler_class(node, self.initializers)
                handler.apply_constraints(self.model, self.variables)

    def get_gurobi_model(self):
        # print(self.variables['input'][0])
        return self.model
