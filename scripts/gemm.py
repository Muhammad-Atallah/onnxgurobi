import numpy as np
from gurobipy import quicksum

class GemmOperator:
    def __init__(self, node, initializers):
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]
        self.input3 = node["input"][2]["name"] if len(node["input"]) > 2 else None
        self.output = node["output"][0]["name"]
        self.initializers = initializers

    def apply_constraints(self, gurobi_model, variables):
        weights = self.initializers.get(self.input2)
        if weights is None:
            raise ValueError(f"Initializer for {self.input2} not found.")

        bias = self.initializers.get(self.input3, np.zeros(weights.shape[1]))

        var_input = variables[self.input1]
        var_output = variables[self.output]

        gurobi_model.update()

        input_size = len(var_input)
        output_size = len(var_output)

        # Checking if weights need to be transposed
        if weights.shape != (input_size, output_size):
            if weights.shape == (output_size, input_size):
                weights = weights.T  # Transposing the weights
            else:
                raise ValueError(f"Unexpected weights shape {weights.shape}")

        for i in range(output_size):
            expression = quicksum(weights[j, i] * var_input[j] for j in range(input_size)) + bias[i]
            gurobi_model.addConstr(var_output[i] == expression, name=f"Gemm_{self.output}_{i}")
