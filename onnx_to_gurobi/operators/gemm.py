import numpy as np
from gurobipy import quicksum
from itertools import product
from .base_operator import BaseOperator
from ..utils import _node_to_string
class GemmOperator(BaseOperator):
    """
    Implements the Gemm operator (General Matrix Multiplication).

    Attributes:
        name (str): The name of this node in the ONNX graph.
        input1 (str): The name of the primary input tensor.
        input2 (str): The name of the second input (weights).
        input3 (str): Optional name of the third input (bias), if present.
        output (str): The name of the output tensor.
        input1_shape (list): Shape of the primary input.
        input2_shape (list): Shape of the second input (weights).
        output_shape (list): Shape of the output.
        initializers (dict): A dictionary of initial values for any constant tensors.

    """

    def __init__(self, node, initializers):
        """
        Initializes the Gemm operator with the node and initializers.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        """
        super().__init__(node, initializers)
        self.name = node["name"]
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]
        self.input3 = node["input"][2]["name"] if len(node["input"]) > 2 else None
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.input2_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = initializers

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to encode the Gemm operation.

        This method represents a matrix multiplication plus optional bias addition.
        It checks if the second input (weights) needs to be transposed and sums
        over the shared dimension to produce each output element. The bias is then
        added to each element.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model in which constraints are created.
            variables (dict): A dictionary mapping tensor names to Gurobi variables or constant values.


        Raises:
            ValueError: If input, output, or weights cannot be found,
            or the weight's shape does not match the expected dimensions.
            IndexError: If an output index exceeds the bounds of the weight matrix.
        """
        weights = self.initializers.get(self.input2)
        bias = self.initializers.get(self.input3, np.zeros(weights.shape[1]))
        var_input = variables[self.input1]
        var_output = variables[self.output]
        var_input_shape = self.input1_shape
        var_output_shape = self.output_shape

        if var_input is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.input}' not found."
            )
        if var_output is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Variable for input '{self.output}' not found."
            )
        if weights is None:
            raise ValueError(
                f"Error in {_node_to_string(self.node)}:"
                f"Initializer for {self.input2} not found."
            )

        gurobi_model.update()

        # Checking if weights need to be transposed
        if weights.shape[0] != var_input_shape[-1]:
            if weights.shape[-1] == var_input_shape[-1]:
                weights = weights.T  # Transpose the weights
            else:
                raise ValueError(
                    f"Error in {_node_to_string(self.node)}:"
                    f"Unexpected weights shape {weights.shape}"
                )

        # Get the common dimension size for summation
        sum_dim = var_input_shape[-1]

        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:
            if len(idx) > 1:
                batch_indices = idx[:-1]
            else:
                batch_indices = ()
            output_idx = idx

            # Ensure the last element is within bounds
            if idx[-1] >= weights.shape[1]:
                raise IndexError(
                    f"Error in {_node_to_string(self.node)}:"
                    f"Index {idx[-1]} is out of bounds for axis 1 with shape {weights.shape[1]} "
                )

            expression = quicksum(
                var_input[batch_indices + (k,)] * float(weights[k, idx[-1]])
                for k in range(sum_dim)
            ) + float(bias[idx])

            gurobi_model.addConstr(
                var_output[output_idx] == expression,
                name=f"Gemm_{self.output}_{'_'.join(map(str, idx))}"
            )
