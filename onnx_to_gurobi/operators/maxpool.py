from gurobipy import GRB
from gurobipy import quicksum
from itertools import product
from base_operator import BaseOperator

class MaxPoolOperator(BaseOperator):
    def __init__(self, node, initializers):
        super().__init__(node, initializers)
        self.input = node["input"][0]["name"]
        self.output = node["output"][0]["name"]
        self.input_shape = node["input"][0]["shape"]
        self.output_shape = node["output"][0]["shape"]

        self.kernel_shape = node["attributes"].get('kernel_shape', [1, 1])
        self.pads = node["attributes"].get('pads', [0, 0, 0, 0])
        self.strides = node["attributes"].get('strides', [1, 1])
        self.dilations = node["attributes"].get('dilations', [1, 1])
        self.ceil_mode = node["attributes"].get('ceil_mode', 0)

    def apply_constraints(self, gurobi_model, variables):
        var_input = variables.get(self.input)
        var_output = variables.get(self.output)

        if var_input is None:
            raise ValueError(f"Variable for input '{self.input}' not found.")
        if var_output is None:
            raise ValueError(f"Variable for output '{self.output}' not found.")

        batch_size = 1
        channels, height_in, width_in = self.input_shape
        channels, height_out, width_out = self.output_shape
        kernel_height, kernel_width = self.kernel_shape
        stride_h, stride_w = self.strides
        pad_top, pad_left, pad_bottom, pad_right = self.pads
        dilation_h, dilation_w = self.dilations

        for c in range(channels):
            for h in range(height_out):
                for w in range(width_out):
                    h_start = h * stride_h - pad_top
                    w_start = w * stride_w - pad_left
                    h_end = h_start + (kernel_height - 1) * dilation_h + 1
                    w_end = w_start + (kernel_width - 1) * dilation_w + 1

                    pooling_elements = []

                    for kh in range(kernel_height):
                        for kw in range(kernel_width):
                            h_in = h_start + kh * dilation_h
                            w_in = w_start + kw * dilation_w

                            if 0 <= h_in < height_in and 0 <= w_in < width_in:
                                pooling_elements.append(var_input[c, h_in, w_in])

                    # Add constraints to ensure var_output[n, c, h, w] is the maximum of pooling_elements

                    # var_output <= each pooling element
                    for idx, elem in enumerate(pooling_elements):
                        gurobi_model.addConstr(
                            var_output[c, h, w] <= elem,
                            name=f"MaxPool_{self.output}_1_{c}_{h}_{w}_upper_{idx}"
                        )

                    # Introduce binary variables to enforce var_output >= one of the pooling elements
                    binary_vars = gurobi_model.addVars(len(pooling_elements), vtype=GRB.BINARY, name=f"MaxPool_bin_{self.output}_1_{c}_{h}_{w}")
                    upper_bound = 1e10

                    # var_output >= pooling_element - M*(1 - binary_var)
                    for idx, elem in enumerate(pooling_elements):
                        gurobi_model.addConstr(
                            var_output[c, h, w] >= elem - upper_bound * (1 - binary_vars[idx]),
                            name=f"MaxPool_{self.output}_1_{c}_{h}_{w}_lower_{idx}"
                        )

                    # Ensure at least one binary_var is 1
                    gurobi_model.addConstr(
                        quicksum(binary_vars[idx] for idx in range(len(pooling_elements))) >= 1,
                        name=f"MaxPool_{self.output}_1_{c}_{h}_{w}_binary_sum"
                    )
