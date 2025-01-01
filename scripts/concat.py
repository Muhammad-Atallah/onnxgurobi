import numpy as np
from gurobipy import GRB
from itertools import product
from base_operator import BaseOperator


class ConcatOperator(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.inputs = [input['name'] for input in node["input"]]  # inputs to concatenate
        self.output = node["output"][0]['name']
        self.inputs_shapes = [input['shape'] for input in node["input"]]
        self.output_shape = node["output"][0]['shape']
        self.axis = None  # Extracting the axis, default to 0 if not specified
        for attr in node.get("attributes", []):
            if attr["name"] == "axis":
                self.axis = attr["value"]
                break
        if self.axis is None:
            self.axis = 0  # Default axis

    def apply_constraints(self, gurobi_model, variables):
        input_vars = [variables[input_name] for input_name in self.inputs]
        output_vars = variables[self.output]
        input_vars_shapes = self.inputs_shapes
        output_vars_shape = self.output_shape

        current_offset = 0
        for input_var, input_shape in zip(input_vars, self.inputs_shapes):
            dim = input_shape[0]  # Since we're concatenating on the first dimension
            for i in range(dim):
                # Assuming other dimensions are the same
                for other_indices in product(*[range(s) for s in input_shape[1:]]):
                    full_output_index = (current_offset + i,) + other_indices
                    full_input_index = (i,) + other_indices

                    gurobi_model.addConstr(
                        output_vars[full_output_index] == input_var[full_input_index],
                        name=f"Concat_{self.output}_{'_'.join(map(str, full_output_index))}"
                    )
            current_offset += dim
