# main.py

import onnx
import gurobipy
from onnxToGurobi import ONNXToGurobi
from parser import ONNXParser
from gemm import GemmOperator
from gurobipy import GRB

def main():
    onnx_model_path = "simple_dd.onnx"
    converter = ONNXToGurobi(onnx_model_path)
    converter.build_model()

    # Prepare the input data
    import numpy as np
    dummy_input = np.random.randn(1, 28*28).astype(np.float32)

    # Retrieve the Gurobi model and variables
    gurobi_model = converter.get_gurobi_model()
    input_tensor_name = 'input'
    input_vars = converter.variables.get(input_tensor_name)

    if input_vars is None:
        print(f"No variables found for input tensor '{input_tensor_name}'.")
    else:
        # Flatten the input data
        input_data = dummy_input.flatten()

        # Fix each input variable by setting bounds
        for idx in input_vars.keys():
            var = input_vars[idx]
            value = float(input_data[idx])
            var.lb = value
            var.ub = value

    # Optimize the model
    gurobi_model.optimize()

    # After optimization attempt
    if gurobi_model.status == GRB.INFEASIBLE:
        print("Model is infeasible; computing IIS...")
        gurobi_model.computeIIS()
        gurobi_model.write("model.ilp")
        print("IIS written to 'model.ilp'")

        # Optionally, apply feasibility relaxation
        print("Applying feasibility relaxation...")
        gurobi_model.feasRelaxS(0, False, False, True)
        gurobi_model.optimize()

        if gurobi_model.status == GRB.OPTIMAL:
            print("Feasibility relaxation found a solution.")
            # Proceed to extract outputs
        else:
            print("Feasibility relaxation did not find a solution.")
    else:
        # Proceed as before
        if gurobi_model.status == GRB.OPTIMAL:
            print("Optimization was successful. Extracting outputs...")
            # Proceed to extract outputs

        # Check if the optimization was successful
        if gurobi_model.status == GRB.OPTIMAL:
            print("Optimization was successful. Extracting outputs...")
        else:
            print(f"Optimization ended with status {gurobi_model.status}.")
            return

    # Extract and print the outputs
    output_tensor_name = 'output'  # Replace with your actual output tensor name
    output_vars = converter.variables.get(output_tensor_name)

    if output_vars is None:
        print(f"No variables found for output tensor '{output_tensor_name}'.")
    else:
        print(f"Outputs for tensor '{output_tensor_name}':")
        gurobi_outputs = []
        for idx in sorted(output_vars.keys()):
            var = output_vars[idx]
            gurobi_outputs.append(var.x)
            print(f"  {output_tensor_name}[{idx}] = {var.x}")

    # Run inference with ONNX Runtime for comparison
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model_path)
    onnx_outputs = session.run(None, {'input': dummy_input})

    # Extract ONNX output
    onnx_output = onnx_outputs[0].flatten()

    # Compare the outputs
    print("\nComparison of outputs:")
    for idx in range(len(onnx_output)):
        print(f"Index {idx}: ONNX Output = {onnx_output[idx]}, Gurobi Output = {gurobi_outputs[idx]}")

if __name__ == "__main__":
    main()
