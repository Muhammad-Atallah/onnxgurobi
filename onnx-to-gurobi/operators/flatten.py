from gurobipy import GRB
from itertools import product
import numpy as np
from base_operator import BaseOperator


class FlattenOperator(BaseOperator):
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
        # Calculate total number of elements for input and output
        input_total = np.prod(var_input_shape)
        output_total = np.prod(var_output_shape)

        if input_total != output_total:
            raise ValueError(f"Total elements mismatch: input has {input_total}, output has {output_total}.")

        # Generate all multi-dimensional indices for the input tensor
        input_indices = list(product(*[range(dim) for dim in var_input_shape]))

        # Map each input index to a flat output index
        for flat_idx, multi_idx in enumerate(input_indices):
            constraint_name = f"Flatten_{self.output}_{flat_idx}"
            gurobi_model.addConstr(
                var_output[flat_idx,] == var_input[multi_idx],
                name=constraint_name
            )
