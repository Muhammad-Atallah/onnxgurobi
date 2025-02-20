import numpy as np
import onnxruntime as ort
from gurobipy import GRB
from onnx_to_gurobi.onnxToGurobi import ONNXToGurobi

def run_onnx_model(model_path, input_data, input_tensor_name='input'):
    """
    Runs an ONNX model with the given input and returns its first output.
    """
    session = ort.InferenceSession(model_path)
    onnx_outputs = session.run(None, {input_tensor_name: input_data})
    return onnx_outputs[0]

def solve_gurobi_model(model_path, input_data, input_tensor_name='input', output_tensor_name='output'):
    """
    Converts an ONNX model to a Gurobi model, assigns input values, optimizes, and returns the output.
    """
    converter = ONNXToGurobi(model_path)
    converter.build_model()

    dummy_input = input_data
    input_shape = dummy_input.shape

    # Set dummy input values in the Gurobi model.
    input_vars = converter.variables.get(input_tensor_name)
    if input_vars is None:
        raise ValueError(f"No input variables found for '{input_tensor_name}'.")
    
    for idx, var in input_vars.items():
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, input_shape[1:])  # Exclude batch dimension
        elif isinstance(idx, tuple):
            if len(idx) < len(input_shape) - 1:
                idx = (0,) * (len(input_shape) - 1 - len(idx)) + idx
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type: {type(idx)}")
        value = float(dummy_input[0, *md_idx])
        var.lb = value
        var.ub = value

    gurobi_model = converter.get_gurobi_model()
    gurobi_model.optimize()
    if gurobi_model.status != GRB.OPTIMAL:
        raise ValueError(f"Optimization ended with status {gurobi_model.status}.")

    # Extract the output from the Gurobi model.
    output_vars = converter.variables.get(output_tensor_name)
    if output_vars is None:
        raise ValueError(f"No output variables found for '{output_tensor_name}'.")
    output_shape = converter.in_out_tensors_shapes[output_tensor_name]
    gurobi_outputs = np.zeros([1] + output_shape, dtype=np.float32)
    
    for idx, var in output_vars.items():
        if isinstance(idx, int):
            md_idx = np.unravel_index(idx, output_shape)
        elif isinstance(idx, tuple):
            md_idx = idx
        else:
            raise ValueError(f"Unexpected index type in output: {type(idx)}")
        gurobi_outputs[(0,) + md_idx] = var.x

    return gurobi_outputs

def compare_models(model_path, input_data, input_tensor_name='input', output_tensor_name='output', atol=1e-5):
    """
    Runs both the ONNX model and the Gurobi model, then asserts that their outputs are close.
    """
    onnx_output = run_onnx_model(model_path, input_data, input_tensor_name)
    gurobi_output = solve_gurobi_model(model_path, input_data, input_tensor_name, output_tensor_name)
    
    if onnx_output.shape != gurobi_output.shape:
        raise ValueError(f"Shape mismatch: ONNX {onnx_output.shape} vs Gurobi {gurobi_output.shape}")
    
    np.testing.assert_allclose(onnx_output, gurobi_output, atol=atol)
