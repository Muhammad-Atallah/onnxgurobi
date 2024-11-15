import numpy as np
from gurobipy import quicksum

class FlattenOperator:
    def __init__(self, node):
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]

    def apply_constraints(self, gurobi_model, variables):
        var_input = variables[self.input]
        var_output = variables[self.output]
        output_size = len(var_output)

        for i in range(output_size):
            gurobi_model.addConstr(var_output[i] == var_input[i], name=f"Reshape_{self.output}_{i}")