import struct
import numpy as np
from base_parser import BaseParser
from utils import _get_data_type

class ConstantParser(BaseParser):
    def parse(self, node, parser):
        outputs = []
        attributes = []
        scalar_data_type = node.attribute[0].t.data_type  # Data type as integer code
        scalar_raw_data = node.attribute[0].t.raw_data    # Raw binary data
        dims = node.attribute[0].t.dims                   # Tensor dimensions

        if len(dims) != 0:
            dims = [int(dim) for dim in dims]
            num_elements = int(np.prod(dims))  # Cast to int
            format_char = _get_data_type(scalar_data_type)

            # Calculate expected bytes
            bytes_per_element = struct.calcsize(format_char)
            expected_bytes = bytes_per_element * num_elements
            if len(scalar_raw_data) != expected_bytes:
                raise ValueError(f"Expected {expected_bytes} bytes for Constant node '{node.name}', but got {len(scalar_raw_data)} bytes.")

            # Unpack and reshape
            values_flat = struct.unpack(format_char * num_elements, scalar_raw_data)
            reshaped_values = np.reshape(values_flat, dims).tolist()

            for out in node.output:
                parser.constant_values[out] = reshaped_values
                outputs.append({'name': out, 'shape': list(dims)})
                parser.intermediate_tensors_shapes[out] = list(dims)
                attributes.append({'name': 'value', 'value': reshaped_values})

        else:
            for out in node.output:
                outputs.append({'name': out, 'shape': 1})
                parser.intermediate_tensors_shapes[out] = 1
                parser.constant_values[out] = struct.unpack(_get_data_type(scalar_data_type), scalar_raw_data)[0]
            for attribute in node.attribute:
                attributes.append({
                    'name': attribute.name,
                    'value': struct.unpack(_get_data_type(scalar_data_type), scalar_raw_data)[0]
                })

        parser.nodes.append({
            'name': node.name,
            'type': node.op_type,
            'input': [],
            'output': outputs,
            'attributes': attributes,
            'initializers': parser.initializer_values,
            'constants': parser.constant_values
        })
