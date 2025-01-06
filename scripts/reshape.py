from gurobipy import GRB
from itertools import product
import numpy as np
from base_operator import BaseOperator

class ReshapeOperator(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]

    def apply_constraints(self, gurobi_model, variables):
        var_input = variables.get(self.input)
        var_output = variables.get(self.output)
        var_input_shape = self.input_shape
        var_output_shape = self.output_shape

        if var_input is None:
            raise ValueError(f"Variable for input '{self.input}' not found.")
        if var_output is None:
            raise ValueError(f"Variable for output '{self.output}' not found.")

        gurobi_model.update()

        # Total number of elements
        input_total = np.prod(var_input_shape)
        output_total = np.prod(var_output_shape)

        if input_total != output_total:
            raise ValueError(f"Total elements mismatch: input has {input_total}, output has {output_total}.")

        # Generate all indices for the output tensor
        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            # Compute the linear index for the current output index
            linear_idx = np.ravel_multi_index(idx, var_output_shape)

            # Convert the linear index to the corresponding input index
            input_idx = np.unravel_index(linear_idx, var_input_shape)

            constraint_name = f"Reshape_{self.output}_{'_'.join(map(str, idx))}"

            gurobi_model.addConstr(
                var_output[idx] == var_input[input_idx],
                name=constraint_name
            )
