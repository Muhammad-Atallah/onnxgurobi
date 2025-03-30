import numpy as np
from onnx_to_gurobi.onnxToGurobi import ONNXToGurobi
from gurobipy import GRB
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1) Load MNIST sample, display a sample image and its correct label
dataset = mnist.load_data()
images, labels = dataset[1]
image = images[0]
label = labels[0]
image_array = np.array(image, dtype=np.float32) / 255.0  # Convert image to a NumPy array and normalize to [0, 1]

plt.imshow(image_array, cmap="gray")
plt.title(f"Original Image | True label: {label}")
plt.axis("off")
plt.show()

# 2) Define a slight perturbation of an input image
eps = 0.3 # Each pixel is allowed to be changed by the value epsilon.
delta = 0.01 # A small margin delta to enforce misclassification

# 3) Convert the ONNX model to a Gurobi model
model_builder = ONNXToGurobi("./onnxgurobi/examples/mnist_classifier.onnx")
model_builder.build_model()
gurobi_model = model_builder.get_gurobi_model()

# 4) Provide input data with defined perturbation
input_vars = model_builder.variables.get("input")
if input_vars is None:
    raise ValueError("No variables found for input tensor.")

input_image_array = image_array.reshape(1, 1, 28, 28).astype(np.float32)
input_shape = input_image_array.shape
for idx, var in input_vars.items():
    if isinstance(idx, int):
        md_idx = np.unravel_index(idx, input_shape[1:])  # Exclude batch dimension
    else:
        md_idx = idx
    original_value = float(input_image_array[0, *md_idx])
    lb = max(0.0, original_value - eps)
    ub = min(1.0, original_value + eps)
    gurobi_model.addConstr(var >= lb, name=f"input_lb_{idx}")
    gurobi_model.addConstr(var <= ub, name=f"input_ub_{idx}")

# 5) Add misclassification constraint
output_vars = model_builder.variables.get("output")
if output_vars is None:
    raise ValueError("No variables found for output tensor.")
for idx, var in output_vars.items():
    if isinstance(idx, tuple):
        class_label = idx[0]
    else:
        class_label = idx
    if class_label != label:
        gurobi_model.addConstr(
            output_vars[(label,)] <= var - delta,
            name=f"misclassify_{class_label}"
        )

# 6) Optimize the model
gurobi_model.optimize()
if gurobi_model.status != GRB.OPTIMAL:
    raise ValueError(f"Model could not be optimized. Status: {gurobi_model.status}")

# 7) Retrieve outputs and display results
output_shape = model_builder.in_out_tensors_shapes["output"]
model_output = np.zeros((1,) + tuple(output_shape), dtype=np.float32)
for idx, var in output_vars.items():
    if isinstance(idx, int):
        md_idx = np.unravel_index(idx, output_shape)
    else:
        md_idx = idx
    model_output[(0,) + md_idx] = var.x

predicted_label = np.argmax(model_output[0])

adversarial_input = np.zeros_like(input_image_array, dtype=np.float32)
for idx, var in input_vars.items():
    if isinstance(idx, int):
        md_idx = np.unravel_index(idx, input_shape[1:])
    else:
        md_idx = idx
    adversarial_input[(0,) + md_idx] = var.x

plt.imshow(adversarial_input[0, 0], cmap="gray")
plt.title(f"True label: {label} | Predicted label: {predicted_label}")
plt.axis("off")
plt.show()