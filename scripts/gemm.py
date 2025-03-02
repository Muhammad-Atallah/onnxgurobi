import numpy as np
from gurobipy import quicksum
from itertools import product
from base_operator import BaseOperator

class GemmOperator(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.name = node["name"]
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]
        self.input3 = node["input"][2]["name"] if len(node["input"]) > 2 else None
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.input2_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = initializers
        self.attributes = node["attributes"]

    def apply_constraints(self, gurobi_model, variables):
        weights = self.initializers.get(self.input2)
        bias = self.initializers.get(self.input3, np.zeros(weights.shape[1]))
        alpha =  self.attributes[0]['value']
        beta =  self.attributes[0]['value']
        print("THIS IS ALPHA:", alpha)
        print("THIS IS BETA:", beta)
        var_input = variables[self.input1]
        var_output = variables[self.output]
        var_input_shape = self.input1_shape
        var_output_shape = self.output_shape

        if weights is None:
            raise ValueError(f"Initializer for {self.input2} not found.")

        gurobi_model.update()

        # Checking if weights need to be transposed
        if weights.shape[0] != var_input_shape[-1]:
            if weights.shape[-1] == var_input_shape[-1]:
                weights = weights.T  # Transpose the weights
            else:
                raise ValueError(f"Unexpected weights shape {weights.shape}")

        # Get the common dimension size for summation
        sum_dim = var_input_shape[-1]

        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            if len(idx) > 1:
                batch_indices = idx[:-1]
            else:
                batch_indices = ()
            output_idx = idx
            # Ensure last element is within bounds
            if idx[-1] >= weights.shape[1]:
                raise IndexError(f"Index {idx[-1]} is out of bounds for axis 1 with size {weights.shape[1]} for node {self.name}")

            expression = quicksum(alpha * var_input[batch_indices + (k,)] * float(weights[k, idx[-1]]) for k in range(sum_dim)) + float(bias[idx])

            gurobi_model.addConstr(var_output[output_idx] == expression, name=f"Gemm_{self.output}_{'_'.join(map(str, idx))}")

