import numpy as np
from gurobipy import GRB

class ConcatOperator:
    def __init__(self, node):
        self.inputs = [input['name'] for input in node["input"]] # inputs to concatenate
        self.output = node["output"][0]["name"]
        self.axis = None            #extracting the axis just in case, don't think I need it now
        for attr in node["attributes"]:
            if attr["name"] == "axis":
                self.axis = attr["value"]
                break
        if self.axis is None:
            self.axis = 0

    def apply_constraints(self, gurobi_model, variables):
        input_vars = [variables[input_name] for input_name in self.inputs]
        output_vars = variables[self.output]

        current_index = 0   #for the output
        for input_var in input_vars:
            input_length = len(input_var)
            for i in range(input_length):
                output_index = current_index + i    #starting always with the current index and increasing output index by 1 until the length of this exact input is exhausted
                gurobi_model.addConstr(
                    output_vars[output_index] == input_var[i],
                    name=f"Concat_{self.output}_{output_index}"
                )
            current_index += input_length # output length = all input lengths added together
