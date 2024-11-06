from gurobipy import GRB

class ReLUOperator:
    def __init__(self, node, initializers):
        self.input = node.input[0]
        self.output = node.output[0]

    def apply_constraints(self, gurobi_model, variables):
        var_input = variables[self.input]
        var_output = variables[self.output]
        upper_bound = pow(10, 10)

        binary_var = gurobi_model.addVars(var_input.keys(), vtype=GRB.BINARY, name=f"relu_binary_{self.output}")

        for idx in var_input.keys():
            gurobi_model.addConstr(var_output[idx] >= var_input[idx])
            gurobi_model.addConstr(var_output[idx] >= 0)
            gurobi_model.addConstr(var_output[idx] <= upper_bound)
            gurobi_model.addConstr(var_output[idx] <= var_input[idx] + upper_bound * (1 - binary_var[idx]))
            gurobi_model.addConstr(var_output[idx] <= upper_bound * binary_var[idx])