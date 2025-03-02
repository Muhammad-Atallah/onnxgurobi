# main.py

import onnx
import gurobipy
# from onnxToGurobi import ONNXToGurobi
from model_builder import ONNXToGurobi
from gurobipy import GRB

def main():
    onnx_model_path = "simple_add.onnx"
    converter = ONNXToGurobi(onnx_model_path)
    converter.build_model()

    # Prepare the input data
    import numpy as np
    dummy_input = np.random.randn(1, 28*28).astype(np.float32)
    input_shape = dummy_input.shape  # (1, 1, 4, 4)

    # Retrieve the Gurobi model and variables
    gurobi_model = converter.get_gurobi_model()
    input_tensor_name = 'input'  # Ensure this matches your actual input tensor name
    input_vars = converter.variables.get(input_tensor_name)

    if input_vars is None:
        print(f"No variables found for input tensor '{input_tensor_name}'.")
        return

    num_dims = len(input_shape)

    # Assign input values to Gurobi variables
    for idx in input_vars.keys():
        var = input_vars[idx]
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, input_shape[1:])  # Exclude batch dimension
        elif isinstance(idx, tuple):
            # Ensure idx has the same number of dimensions as input_shape (excluding batch)
            if len(idx) < num_dims - 1:
                # Prepend zeros for missing dimensions (e.g., batch dimension)
                idx = (0,) * (num_dims - 1 - len(idx)) + idx
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type: {type(idx)}")
        # Assign the corresponding value from dummy_input to the Gurobi variable
        value = float(dummy_input[0, *md_idx])  # Fixed batch index
        var.lb = value
        var.ub = value

    # Optimize the model
    gurobi_model.optimize()

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
        return

    # Get the shape of the output tensor from the intermediate tensors
    output_shape = converter.parser.input_output_tensors_shapes[output_tensor_name]  # Correct shape: [4, 4, 4]

    # Initialize gurobi_outputs with batch dimension
    gurobi_outputs = np.zeros([1] + output_shape)  # Shape: [1, 4, 4, 4]

    # Assign Gurobi variable values to gurobi_outputs
    for idx in output_vars.keys():
        var = output_vars[idx]
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, output_shape)
        elif isinstance(idx, tuple):
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type in output_vars: {type(idx)}")

        # Assign the variable's value to the correct position in gurobi_outputs
        gurobi_outputs[0, *md_idx] = var.x

    # Extract ONNX output using ONNX Runtime
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model_path)
    onnx_outputs = session.run(None, {'input': dummy_input})

    # Extract the first (and only) output
    onnx_output = onnx_outputs[0]  # Shape: (1, 4, 4, 4)

    # Ensure gurobi_outputs and onnx_output have the same shape
    if gurobi_outputs.shape != onnx_output.shape:
        print(f"Shape mismatch: gurobi_outputs shape {gurobi_outputs.shape} vs onnx_output shape {onnx_output.shape}")
        return

    print(f"gurobi_outputs shape after initialization: {gurobi_outputs.shape}")
    print(f"onnx_output shape: {onnx_output.shape}")

    # Compare the outputs and print them side by side
    print("\nDetailed comparison of outputs:")
    for idx, onnx_val in np.ndenumerate(onnx_output):
        gurobi_val = gurobi_outputs[idx]
        difference = onnx_val - gurobi_val
        print(f"Index {idx}: ONNX Output = {onnx_val}, Gurobi Output = {gurobi_val}, Difference = {difference}")

    # Calculate overall statistics
    print("\nSummary of differences:")
    differences = np.abs(onnx_output - gurobi_outputs)
    max_difference = np.max(differences)
    mean_difference = np.mean(differences)
    all_close = np.allclose(onnx_output, gurobi_outputs, atol=1e-5)
    print(f"All outputs match within tolerance: {all_close}")
    print(f"Max difference: {max_difference}")
    print(f"Mean difference: {mean_difference}")

if __name__ == "__main__":
    main()
