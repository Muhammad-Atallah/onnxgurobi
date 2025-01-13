import numpy as np
from gurobipy import quicksum
from itertools import product
from base_operator import BaseOperator

class BatchNormalization(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.name = node["name"]
        self.input = node["input"][0]["name"]
        self.weights = node["input"][1]["name"]
        self.bias = node["input"][2]["name"]
        self.mean = node["input"][3]["name"]
        self.variance = node["input"][4]["name"]
        self.output = node["output"][0]["name"]
        self.shape_input_output = node["input"][0]["shape"]
        self.initializers = initializers
        self.epsilon = node["attributes"].get("epsilon", 1e-5)

    def apply_constraints(self, gurobi_model, variables):
        weights = self.initializers.get(self.weights)
        bias = self.initializers.get(self.bias)
        mean = variables[self.mean]
        variance = variables[self.variance]
        var_input = variables[self.input]
        var_output = variables[self.output]

        if var_input is None:
            raise ValueError(f"Variable for input '{self.input}' not found.")
        if var_output is None:
            raise ValueError(f"Variable for output '{self.output}' not found.")

        gurobi_model.update()

        a = weights / np.sqrt(variance + self.epsilon)
        b = bias - (weights * mean) / np.sqrt(variance + self.epsilon)

        gurobi_model.update()

        output_indices = list(product(*[range(dim) for dim in self.shape_input_output]))

        for idx in output_indices:
            channel = idx[0]
            gurobi_model.addConstr(
                var_output[idx] == a[channel] * var_input[idx] + b[channel],
                name=f"BatchNorm_{self.name}_{'_'.join(map(str, idx))}"
            )


