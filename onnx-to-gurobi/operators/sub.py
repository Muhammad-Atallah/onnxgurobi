# onnx_to_gurobi/operators/add.py
from gurobipy import quicksum
import numpy as np

class SubOperator:
    def __init__(self, node):
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]      # Scalar or vector
        self.output = node["output"][0]["name"]
        self.initializers = node["initializers"]

    def apply_constraints(self, gurobi_model, variables):
        var_input1 = self.initializers.get(self.input1)
        if var_input1 is None:
            var_input1 = variables.get(self.input1)
        var_input2 = self.initializers.get(self.input2)
        if var_input2 is None:
            var_input2 = variables.get(self.input2)
        var_output = variables.get(self.output)
        gurobi_model.update()

        if var_input1 is None:
            raise ValueError(f"Variable for input '{self.input1}' not found.")
        if var_input2 is None:
            raise ValueError(f"Variable or constant for input '{self.input2}' not found.")
        if var_output is None:
            raise ValueError(f"Variable for output '{self.output}' not found.")

        # Check if var_input2 is a vector (dict or array)
        if isinstance(var_input2, dict) or isinstance(var_input2, np.ndarray):
            # Both inputs are vectors
            for i in range(len(var_output)):
                gurobi_model.addConstr(
                    var_output[i] == var_input1[i] - var_input2[i],
                    name=f"Sub_{self.output}_{i}"
                )
        else:
            # var_input2 is a scalar
            for i in range(len(var_output)):
                gurobi_model.addConstr(
                    var_output[i] == var_input1[i] - var_input2,
                    name=f"Sub_{self.output}_{i}"
                )
