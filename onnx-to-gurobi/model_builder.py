from gurobipy import Model, GRB
from parser import ONNXParser
from gemm import GemmOperator
from add import AddOperator
from matmul import MatMul

class ONNXToGurobi:
    def __init__(self, onnx_model_path):
        self.parser = ONNXParser(onnx_model_path)
        self.model = Model("NeuralNetwork")
        self.variables = {}
        self.initializers = self.parser.initializer_values
        self.nodes = self.parser.nodes
        self.node_handlers = {
            'Gemm'   : GemmOperator,
            'Add'    : AddOperator,
            'MatMul' : MatMul
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
            if(node['type'] == "Constant"):
                self.variables[output_name] = node['attributes'][0]['value']        #No need to create a variable for the scalar. Just saving the value of the constant here.
            else:
                size = node['output'][0]['shape']
                self.variables[output_name] = self.model.addVars(
                    range(size),
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
                if op_type in ['Add']:
                    handler = handler_class(node)
                else:
                    handler = handler_class(node, self.initializers)
                handler.apply_constraints(self.model, self.variables)

    def get_gurobi_model(self):
        # print(self.variables['input'][0])
        return self.model
