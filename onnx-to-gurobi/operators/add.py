from itertools import product
import numpy as np
from base_operator import BaseOperator

class AddOperator(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]      # Scalar or tensor
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.input2_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = node["initializers"]

    def apply_constraints(self, gurobi_model, variables):
        var_input1 = self.initializers.get(self.input1)
        if var_input1 is None:
            var_input1 = variables.get(self.input1)
        var_input2 = self.initializers.get(self.input2)
        if var_input2 is None:
            var_input2 = variables.get(self.input2)
        var_output = variables.get(self.output)
        var_input1_shape = self.input1_shape
        var_input2_shape = self.input2_shape
        var_output_shape = self.output_shape

        gurobi_model.update()

        if var_input1 is None:
            raise ValueError(f"Variable for input '{self.input1}' not found.")
        if var_input2 is None:
            raise ValueError(f"Variable or constant for input '{self.input2}' not found.")
        if var_output is None:
            raise ValueError(f"Variable for output '{self.output}' not found.")

        # Generate all indices for the output tensor
        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            # Check if input2 is a tensor or a scalar
            if isinstance(var_input2, dict) or isinstance(var_input2, np.ndarray):
                if var_input1_shape != var_input2_shape:
                    raise ValueError(f"Shape mismatch: input1 shape {var_input1_shape} != input2 shape {var_input2_shape}")

                expression = var_input1[idx] + var_input2[idx]
            else:
                # input2 is a scalar
                expression = var_input1[idx] + var_input2

            if isinstance(idx, tuple):
                constraint_name = f"Add_{self.output}_{'_'.join(map(str, idx))}"
            else:
                constraint_name = f"Add_{self.output}_{idx}"

            gurobi_model.addConstr(var_output[idx] == expression, name=constraint_name)
