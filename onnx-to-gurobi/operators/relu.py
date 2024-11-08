from gurobipy import GRB

class ReLUOperator:
    def __init__(self, node, initializers):
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]

    def apply_constraints(self, gurobi_model, variables):
        var_input = variables[self.input]
        var_output = variables[self.output]
        binary_var = variables[f"relu_binary_{self.output}"]
        upper_bound = 1e10

        for i in var_input.keys():
            gurobi_model.addConstr(var_output[i] >= var_input[i])
            gurobi_model.addConstr(var_output[i] >= 0)
            gurobi_model.addConstr(var_output[i] <= upper_bound)
            gurobi_model.addConstr(var_output[i] <= var_input[i] + upper_bound * (1 - binary_var[i]))
            gurobi_model.addConstr(var_output[i] <= upper_bound * binary_var[i])