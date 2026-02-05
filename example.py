from adaline import adaline_neuron
import numpy as np

"""
Simulates the machine (ADALINE Neuron) created by Widrow and Hoff in 1960
"""

# Define the 4x4 matrices for "T" and "J"
T_matrix = np.array([
    [-1, 1, 1, 1],
    [-1, -1, 1, -1],
    [-1, -1, 1, -1],
    [-1, -1, 1, -1]
])

J_matrix = np.array([
    [-1, 1, 1, 1],
    [-1, -1, -1, 1],
    [-1, -1, -1, 1],
    [-1, 1, 1, 1]
])

# Flatten the matrices into vectors
T_vector = T_matrix.flatten()
J_vector = J_matrix.flatten()

# Define the input dataset and labels
inputs = np.array([T_vector, J_vector])
labels = np.array([1, -1])  # 1 for "T", -1 for "J"

# Initialize the ADALINE neuron
neuron = adaline_neuron()
neuron.initialize(input_size=inputs.shape[1])

# Train the neuron
epochs = 100
for epoch in range(epochs):
    for i in range(len(inputs)):
        neuron.train(inputs[i], labels[i])

# Test the neuron
for i, input_vector in enumerate(inputs):
    neuron.x = input_vector
    output = neuron.compute_y()
    prediction = 1 if output > 0 else -1  # Threshold at 0
    print(f"Input {i + 1}: Predicted = {prediction}, Actual = {labels[i]}")
