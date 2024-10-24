import numpy as np
from gurobipy import quicksum

class GemmOperator:
    def __init__(self, node, initializers):
        print("inside init. Node: ", node["input"][0]["name"])
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]
        self.input3 = node["input"][2]["name"] if len(node["input"]) > 2 else None
        self.output = node["output"][0]["name"]
        self.initializers = initializers

    def apply_constraints(self, gurobi_model, variables):
        weights = self.initializers.get(self.input2)

        if weights is None:
            raise ValueError(f"Initializer for {self.input2} not found.")

        bias = self.initializers.get(self.input3, np.zeros(weights.shape[0]))

        var_input = variables[self.input1]
        var_output = variables[self.output]

        for i in range(len(var_input)):
            if weights.ndim == 1:
                weight_row = weights
            else:
                weight_row = weights[i]

            expression = quicksum(float(weight_row[j]) * var_input[j] for j in range(len(var_input))) + float(bias[i])
            gurobi_model.addConstr(var_output[i] == expression, name=f"Gemm_{self.output}_{i}")
