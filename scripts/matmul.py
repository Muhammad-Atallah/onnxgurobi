import numpy as np
from gurobipy import quicksum
from itertools import product
from base_operator import BaseOperator

class MatMul(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.input2_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = initializers
        self.constants = node["constants"]

    def apply_constraints(self, gurobi_model, variables):
        var_input = variables[self.input1]
        var_output = variables[self.output]
        weights = self.initializers.get(self.input2, np.array(self.constants[self.input2]))
        var_input_shape = self.input1_shape
        var_output_shape = self.output_shape

        if weights is None:
            raise ValueError(f"Initializer for {self.input2} not found.")

        gurobi_model.update()

        # Ensure var_input_shape is a list
        if isinstance(var_input_shape, int):
            var_input_shape = [var_input_shape]

        # Ensure var_output_shape is a list
        if isinstance(var_output_shape, int):
            var_output_shape = [var_output_shape]
        # Checking if weights need to be transposed
        if weights.shape[0] != var_input_shape[-1]:
            if weights.shape[-1] == var_input_shape[-1]:
                weights = weights.T  # Transpose the weights
            else:
                raise ValueError(f"Unexpected weights shape {weights.shape}")


        print("weights.shape[0]::::::::", weights.shape[0])
        print("var_output_shape::::::::", var_output_shape)
        # Get the common dimension size
        sum_dim = var_input_shape[-1]

        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:

            if len(idx) > 1:
                batch_indices = idx[:-1]  # All indices except the last one
            else:
                batch_indices = ()
            output_idx = idx

            # Ensure last element is within bounds
            if idx[-1] >= weights.shape[-1]:  # had to change it from weights.shape[1] to weights.shape[-1]
                raise IndexError(f"Index {idx[-1]} is out of bounds for axis {len(weights.shape)-1} with size {weights.shape[-1]}")

            expression = quicksum(
                var_input[batch_indices + (k,)] * float(weights[batch_indices + (k, idx[-1])])
                for k in range(sum_dim)
            )

            gurobi_model.addConstr(
                var_output[output_idx] == expression,
                name=f"MatMul_{self.output}_{'_'.join(map(str, idx))}"
            )
