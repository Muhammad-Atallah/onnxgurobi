from gurobipy import GRB
from itertools import product
import numpy as np
from base_operator import BaseOperator

class ReLUOperator(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]

    def apply_constraints(self, gurobi_model, variables):
        var_input = variables[self.input]
        var_output = variables[self.output]
        var_input_shape = self.input_shape
        var_output_shape = self.output_shape
        binary_var = variables.get(f"relu_binary_{self.output}")
        upper_bound = 1e5

        gurobi_model.update()

        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            constraint_name = f"ReLU_{self.output}_{'_'.join(map(str, idx))}"

            gurobi_model.addConstr(var_output[idx] >= var_input[idx], name=f"{constraint_name}_ge_x")
            gurobi_model.addConstr(var_output[idx] >= 0, name=f"{constraint_name}_ge_0")
            gurobi_model.addConstr(var_output[idx] <= upper_bound, name=f"{constraint_name}_le_upper_bound")
            gurobi_model.addConstr(var_output[idx] <= var_input[idx] + upper_bound * (1 - binary_var[idx]), name=f"{constraint_name}_le_x_plus_upper_bound")
            gurobi_model.addConstr(var_output[idx] <= upper_bound * binary_var[idx], name=f"{constraint_name}_le_upper_bound_binary")
