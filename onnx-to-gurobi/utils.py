def _node_to_string(node):
    name = node.get("name")
    type = node.get("type")
    inputs = ", ".join(f"name: {inp['name']}, shape: {inp.get('shape')}" for inp in node.get("input"))
    outputs = ", ".join(f"name: {out['name']}, shape: {out.get('shape')}" for out in node.get("output"))
    attributes_str = ", ".join(f"{key}: {value}" for key, value in node.get("attributes").items())

    return (
        f"Node(Name: {name}, Type: {type}, "
        f"Inputs: [{inputs}], Outputs: [{outputs}], "
        f"Attributes: {{{attributes_str}}})"
    )

def _extract_shape(tensor):
    shape = [dim.dim_value for dim in tensor.type.tensor_type.shape.dim[1:]]  # Exclude batch size
    return shape if len(shape) > 1 else [shape[0]]

def _get_data_type(scalar_data_type):
    data_types = {
        1: 'f',   # FLOAT
        2: 'B',   # UINT8
        3: 'b',   # INT8
        4: 'H',   # UINT16
        5: 'h',   # INT16
        6: 'i',   # INT32
        7: 'q',   # INT64
        10: 'e',  # FLOAT16
        11: 'd',  # DOUBLE
        12: 'I',  # UINT32
        13: 'Q',  # UINT64
    }
    if scalar_data_type in data_types:
        return data_types[scalar_data_type]
    else:
        raise ValueError(f"Unsupported data type: {scalar_data_type}")

def _unsqueeze_shape(input_shape, axes):
    output_shape = input_shape.copy()
    for axis in sorted(axes):
        if axis < 0:
            axis += len(output_shape) + 1
        output_shape.insert(axis, 1)
    return output_shape