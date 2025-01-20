import unittest
import numpy as np
import onnxruntime as ort
from gurobipy import GRB
from onnx_to_gurobi.model_builder import ONNXToGurobi

def run_onnx_model(model_path, input_data):
    """
    Run an ONNX model using onnxruntime and return the outputs.

    """
    session = ort.InferenceSession(model_path)

    if isinstance(input_data, dict):
        feed_dict = {}
        for i, input_meta in enumerate(session.get_inputs()):
            feed_dict[input_meta.name] = input_data[input_meta.name]
    else:
        input_name = session.get_inputs()[0].name
        feed_dict = {input_name: input_data}

    outputs = session.run(None, feed_dict)
    return outputs

def solve_gurobi_model(model_path, input_data):
    """
    Convert the ONNX model to a Gurobi model, set input variables, solve,
    and retrieve the outputs.

    """
    converter = ONNXToGurobi(model_path)
    converter.build_model()

    input_tensors = converter.parser.graph.input

    if isinstance(input_data, dict):
        for input_info in input_tensors:
            tensor_name = input_info.name
            var = converter.variables[tensor_name]
            data_array = input_data[tensor_name]

            for idx, _ in np.ndenumerate(data_array):
                var[idx].start = float(data_array[idx])
    else:
        tensor_name = input_tensors[0].name
        var = converter.variables[tensor_name]
        for idx, _ in np.ndenumerate(input_data):
            var[idx].start = float(input_data[idx])

    converter.model.setParam('OutputFlag', 0)
    converter.model.optimize()

    outputs = []
    for output_info in converter.parser.graph.output:
        out_name = output_info.name
        out_var = converter.variables[out_name]
        out_shape = converter.parser.input_output_tensors_shapes[out_name]

        out_array = np.zeros(out_shape, dtype=np.float32)
        for idx in np.ndindex(*out_shape):
            out_array[idx] = out_var[idx].X
        outputs.append(out_array)

    return outputs

def compare_models(model_path, input_data, rtol=1e-4, atol=1e-4):
    """
    Run both the ONNX and Gurobi models and compare outputs within a tolerance.

    """
    # Run model with ONNX Runtime
    onnx_outputs = self.run_onnx_model(model_path, input_data)

    # Run model with Gurobi
    gurobi_outputs = self.solve_gurobi_model(model_path, input_data)

    # Compare each output
    for i in range(len(onnx_outputs)):
        np.testing.assert_allclose(
            onnx_outputs[i],
            gurobi_outputs[i],
            rtol=rtol,
            atol=atol,
            err_msg=f"Mismatch in output {i} for model: {model_path}"
        )
