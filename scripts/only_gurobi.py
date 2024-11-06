# main.py

import onnx
import gurobipy
from model_builder import ONNXToGurobi
from parser import ONNXParser
from gemm import GemmOperator
from gurobipy import GRB
from gurobipy import quicksum


def main():
    onnx_model_path = "D:/Informatik Studium/8. Semester/Bachelor's Thesis/Pytorch/ONNX/only_gemm.onnx"
    converter = ONNXToGurobi(onnx_model_path)
    converter.build_model()

    # Retrieve the Gurobi model
    gurobi_model = converter.get_gurobi_model()

    # Set an objective if necessary
    # For example, minimize the sum of the outputs
    output_vars = converter.variables.get("output")
    if output_vars is not None:
        gurobi_model.setObjective(quicksum(output_vars[i] for i in output_vars), GRB.MINIMIZE)
    else:
        print("Output variables not found.")
        return

    # Optimize the model
    gurobi_model.optimize()

    # Check if the optimization was successful
    if gurobi_model.status == GRB.OPTIMAL:
        print("Optimization was successful. Extracting outputs...")
    else:
        print(f"Optimization ended with status {gurobi_model.status}.")
        return

    # Extract and print the outputs
    print(f"Outputs for tensor 'output':")
    for idx in sorted(output_vars.keys()):
        var = output_vars[idx]
        print(f"  output[{idx}] = {var.X}")

if __name__ == "__main__":
    main()

