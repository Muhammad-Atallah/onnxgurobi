import numpy as np
from gurobipy import quicksum
from itertools import product
from base_operator import BaseOperator
from utils import _node_to_string

class MatMul(BaseOperator):
    """
    Implements the MatMul (matrix multiplication) operator.

    Attributes:
        input1 (str): The name of the first (left-hand side) input tensor.
        input2 (str): The name of the second (right-hand side) input tensor.
        output (str): The name of the output tensor.
        input1_shape (list): The shape of the first input.
        input2_shape (list): The shape of the second input.
        output_shape (list): The shape of the output.
        initializers (dict): A dictionary of initial values for any constant tensors.
        constants (dict): A dictionary with additional constant values if not found in `initializers`.
    """

    def __init__(self, node, initializers):
        """
        Initializes the MatMul operator with the given node and initializers.

        Args:
            node (dict): A dictionary describing the ONNX node. Expected to have the following keys:
            "name", "type", "input", "output", "attributes", "initializers", and "constants".
            initializers (dict): A dictionary of initial values for any constant tensors (weights, biases, etc.).

        """
        super().__init__(node, initializers)
        self.input1 = node["input"][0]["name"]
        self.input2 = node["input"][1]["name"]
        self.output = node["output"][0]["name"]
        self.input1_shape = node["input"][0]["shape"]
        self.input2_shape = node["input"][1]["shape"]
        self.output_shape = node["output"][0]["shape"]
        self.initializers = initializers
        self.constants = node["constants"]

    def apply_constraints(self, gurobi_model, variables):
        """
        Applies the Gurobi constraints to represent the matrix multiplication operation.

        This method retrieves the first input and the second input from the model or the initializers.
        If necessary, it transposes the second input to match the expected dimensions.
        It then loops through every element in the output tensor shape,
        constructing a summation of products of corresponding input elements and weights.

        Args:
            gurobi_model (gurobipy.Model): The Gurobi model in which constraints are created.
            variables (dict): A dictionary mapping tensor names to Gurobi variables or constant values.

        Raises:
            ValueError: If the second input's initializer or constants data is missing,
                or if the operator's internal shapes are unexpected.
            IndexError: If any dimension in the resulting weights array is out of
                bounds for the required operation.
        """
        var_input = variables[self.input1]
        var_output = variables[self.output]
        print("INITIALIZERS:::::::::::", self.initializers[self.input2])
        weights = self.initializers.get(self.input2, np.array(self.constants[self.input2]))
        if weights is None:
            weights = self.initializers[self.input2]
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
            raise ValueError(f"Initializer for '{self.input2}' not found or is None.")

        gurobi_model.update()

        if isinstance(var_input_shape, int):
            var_input_shape = [var_input_shape]
        if isinstance(var_output_shape, int):
            var_output_shape = [var_output_shape]

        if weights.shape[0] != var_input_shape[-1]:
            if weights.shape[-1] == var_input_shape[-1]:
                weights = weights.T
            else:
                raise ValueError(
                    f"Error in {_node_to_string(self.node)}:"
                    f"Unexpected weights shape {weights.shape} for input2 '{self.input2}'. "
                )

        sum_dim = var_input_shape[-1]

        # Generate all multi-dimensional indices for the input tensor
        output_indices = list(product(*[range(dim) for dim in var_output_shape]))

        for idx in output_indices:

            if idx[-1] >= weights.shape[-1]:
                raise IndexError(
                    f"Error in {_node_to_string(self.node)}:"
                    f"Index {idx[-1]} out of bounds for weights with shape {weights.shape[-1]} "
                )

            expression = quicksum(
                var_input[(k,)] * float(weights[(k, idx[-1])])
                for k in range(sum_dim)
            )

            gurobi_model.addConstr(
                var_output[idx] == expression,
                name=f"MatMul_{self.output}_{'_'.join(map(str, idx))}"
            )
