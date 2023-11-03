import random

# Activation function: ReLU
def relu(x):
    return max(0, x)

# Matrix multiplication
def matmul(A, B):
    if len(A[0]) != len(B):
        raise ValueError(f"Incompatible dimensions for matrix multiplication: {len(A[0])}x{len(B)}")
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(A[0])):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Addition of two vectors
def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]

# Apply activation function element-wise
def apply_activation(func, matrix):
    return [[func(x) for x in row] for row in matrix]

# Forward pass through the neural network
def forward_pass(network, input_data):
    layer_input = [input_data]  # Wrap the input data in a list to make it a single-row matrix
    for idx, layer in enumerate(network['Layers']):
        weights = layer['weights']
        biases = layer['biases']

        # Debug: print the biases to check their structure
        print(f"Layer {idx+1} biases before adding: {biases}")

        # Validate the matrix multiplication dimensions
        if len(layer_input[0]) != len(weights[0]):  # Check against the number of rows in the weight matrix
            raise ValueError(f"Layer {idx+1}: Input length {len(layer_input[0])} does not match weight row count {len(weights[0])}")

        # Perform matrix multiplication
        layer_output = matmul(layer_input, weights)

        # Add biases to each row of the result of matrix multiplication
        layer_output = [add_vectors(row, biases) for row in layer_output]

        # Apply ReLU activation if specified
        if layer['relu']:
            layer_output = apply_activation(relu, layer_output)

        # Set the output as the next layer's input
        layer_input = layer_output

    # Assuming the last layer output is a single row matrix, return the first row as the output vector
    return layer_input[0]

# Parse networks from file
def parse_networks(filename):
    networks = []
    current_network = None

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if parts[0].startswith('Layers'):
                current_network = {'Layers': []}
                networks.append(current_network)
            elif parts[0].startswith('Weights'):
                current_layer = {'weights': eval(parts[1].strip())}
                current_network['Layers'].append(current_layer)
            elif parts[0].startswith('Biases'):
                # Parse biases and ensure it is a list (not a list of lists)
                biases = eval(parts[1].strip())
                if isinstance(biases[0], list):  # If the first element is a list, it's a list of lists
                    biases = biases[0]  # Take the first list to flatten it
                current_layer['biases'] = biases
            elif parts[0].startswith('Relu'):
                current_layer['relu'] = parts[1].strip().lower() == 'true'

    return networks

# Sum of absolute values function
def sum_of_abs(values):
    """Compute the sum of absolute values of a list."""
    return sum(abs(value) for value in values)

# Optimization function
def optimize(network, num_iterations=10000, step_size=0.01):
    # Adjust to match the correct input size by taking the length of the first weight matrix
    input_size = len(network['Layers'][0]['weights'][0])  # Corrected to use the length of the first row
    best_input = [random.uniform(-1, 1) for _ in range(input_size)]
    best_output = forward_pass(network, best_input)
    best_score = sum_of_abs(best_output)

    for _ in range(num_iterations):
        candidate_input = [x + random.uniform(-step_size, step_size) for x in best_input]
        candidate_output = forward_pass(network, candidate_input)
        candidate_score = sum_of_abs(candidate_output)
        if candidate_score < best_score:
            best_input, best_output, best_score = candidate_input, candidate_output, candidate_score

    return best_input

# Main function
def main():
    networks = parse_networks('networks.txt')
    with open('solutions.txt', 'w') as f_out:
        for i, network in enumerate(networks):
            try:
                input_data = optimize(network)  # Assume optimize is efficient and deterministic
                input_string = ','.join(map(str, input_data))
                f_out.write(f"{input_string}\n")
            except ValueError as e:
                print(f"Network {i+1} encountered an error: {e}")

if __name__ == "__main__":
    main()
