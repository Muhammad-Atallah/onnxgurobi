import onnx
from onnx import numpy_helper, ModelProto, NodeProto
import numpy as np
import struct

onnx_model = onnx.load("simple_ff.onnx")

graph = onnx_model.graph

input_output_tensors_shapes = {}

constant_node = {}

# for node in graph.node:
#     if node.op_type == "Concat":
#         print(len(node.input))
#         print("-----------------------------")

tens1 = np.array([1,2])
tens2 = np.array([1,2])

print("tens1 shape: ", list(tens1.shape)[1:] == list(tens2.shape)[1:])
# for input in graph.input:
#     shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim[1:]]
#     print("shape:", len(shape))

# for output in graph.output:
#     input_output_tensors_shapes[output.name] = output.type.tensor_type.shape.dim[1].dim_value



# print("input shape:", input_output_tensors_shapes["input"])
    # if(node.op_type == "Constant"):
    # print(node)
    # print("______________________________________________________")
        # print(node.attribute[0].t.data_type)
        # print(struct.unpack("f", node.attribute[0].t.raw_data)[0])
        # print(onnx.numpy_helper.to_array(node.attribute[0].t.raw_data))

# for initializer in graph.initializer:
#     print(initializer.name, onnx.numpy_helper.to_array(initializer).shape)
#     print("______________________________________________________")

# dict = {"first" : "first_value", "second": "second_value"}
# dict["third"] = "third_value"
# print(isinstance(dict, "dict"))





# inputs = {"name": 2, "shape":1}
# print(inputs.get("name"))


###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

# input_output_tensors_shapes = {}
# initializer_shapes = {}
# initializer_values = {}
# tensors = []

# #filling initializer_shapes and initializer_values
# for initializer in graph.initializer:
#     initializer_shapes[initializer.name] = onnx.numpy_helper.to_array(initializer).shape
#     initializer_values[initializer.name] = onnx.numpy_helper.to_array(initializer)

# # filling input_output_tensors_shapes with a loop for inputs and one for outputs
# for input in graph.input:
#     input_output_tensors_shapes[input.name] = input.type.tensor_type.shape.dim[1].dim_value

# for output in graph.output:
#     input_output_tensors_shapes[output.name] = output.type.tensor_type.shape.dim[1].dim_value

# #forming an object for each node
# for node in graph.node:
#     tensors.append({'name': node.name, 'type' : node.op_type,
#                     'input' : [{'name' : input, 'shape' : initializer_shapes.get(input, input_output_tensors_shapes.get(input))} for input in node.input],
#                     'output' : [{'name' : output, 'shape' : initializer_shapes.get(output, input_output_tensors_shapes.get(output))} for output in node.output],
#                     'attributes' : [{'name' : attribute.name, 'value' : attribute.f} for attribute in node.attribute]})

# #start shape from input shape
# current_shape = input_output_tensors_shapes['input']

# #filling tensors of empty shapes with the appropriate shape
# for tensor in tensors:
#     if tensor['type'] != 'Constant':
#         if tensor['type'] == 'Gemm':
#             tensor['input'][0]['shape'] = current_shape if tensor['input'][0]['shape'] is None else tensor['input'][0]['shape']
#             tensor['output'][0]['shape'] = tensor['input'][2]['shape'] if tensor['output'][0]['shape'] is None else  tensor['output'][0]['shape']
#             current_shape = tensor['input'][2]['shape']
#         else:
#             tensor['input'][0]['shape'] = current_shape if tensor['input'][0]['shape'] is None else tensor['input'][0]['shape']
#             tensor['output'][0]['shape'] = current_shape if tensor['output'][0]['shape'] is None else  tensor['output'][0]['shape']

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################


# for tensor in tensors:
#     print(tensor)
#     print('____________________________________')

# print(initializer_shapes)

# for node in graph.node:
#     print(node)
#     print("----------------------------------")


# tensors = []
# for tensor in graph.node:
#     print(tensor.output)
#     tensors.append ({'tensor_name': tensor.name})

# initializers = {}
# for initializer in graph.initializer:
#     initializers[initializer.name] = onnx.numpy_helper.to_array(initializer).shape

# print(initializers)


# for tensor in tensors:
#     print(tensor)
# for node in graph.node:
#     for i in range(len(node.input)):
#         inputs.append({"input": node.input[i], "type": node.op_type})
#     for i in range(len(node.output)):
#         outputs.append(node.output[i])

# print("inputs: ", inputs)
# print("outputs: ", outputs)

# model_proto = ModelProto()
# with open("D:/Informatik Studium/8. Semester/Bachelor's Thesis/Pytorch/ONNX/mnist_model_with.add.onnx", "rb") as file:
#     content = file.read()

# model_proto.ParseFromString(content)

# print(model_proto.graph.initializer)
# print(model_proto.graph.node)

# print(initializers)

# for input in model_proto.graph.input:
#     print("input:", input)
    # print(input.type.tensor_type.shape.dim)

# for input in model_proto.graph.input:
#     print(input.type.tensor_type.shape.dim[1].dim_value)

# for node in onnx_model.graph.node:
#     print(node)
#     print("---------------------------------------------")

# for element in enumerate(initializers):
#     print(element)

# print(initializers)

# print(onnx_model.graph.initializer)

# print(onnx_model.graph.initializer.get(onnx_model.graph.node[5].input[0]))



# print(chr(a[7]), end=" ")
# print(chr(a[9]), end=" ")

