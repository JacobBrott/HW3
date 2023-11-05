import random

# Activation function: ReLU
def relu(x):
    return max(0, x)

# Apply ReLU activation to a vector
def apply_relu(vector):
    return [relu(x) for x in vector]

# Vector dot product
def vector_dot_product(vec_a, vec_b):
    return sum(a * b for a, b in zip(vec_a, vec_b))

# Matrix-vector multiplication
def matrix_vector_multiply(matrix, vector):
    return [vector_dot_product(row, vector) for row in matrix]

# Vector addition
def vector_add(vec_a, vec_b):
    return [a + b for a, b in zip(vec_a, vec_b)]

# Forward pass through the neural network
def forward_pass(network, input_data):
    layer_input = input_data
    for layer in network['Layers']:
        weights = layer['weights']
        biases = layer['biases']
        layer_output = vector_add(matrix_vector_multiply(weights, layer_input), biases)
        if layer.get('relu', False):
            layer_output = apply_relu(layer_output)
        layer_input = layer_output
    return layer_input

# Compute the gradient numerically
def compute_gradient(network, input_data):
    epsilon = 1e-4
    gradient = [0 for _ in input_data]
    for i in range(len(input_data)):
        input_data_epsilon = input_data[:]  # Copy the input data
        input_data_epsilon[i] += epsilon
        output_plus_epsilon = forward_pass(network, input_data_epsilon)
        input_data_epsilon[i] -= 2 * epsilon
        output_minus_epsilon = forward_pass(network, input_data_epsilon)
        gradient[i] = (sum(abs(opl - opm) for opl, opm in zip(output_plus_epsilon, output_minus_epsilon)) / (2 * epsilon))
    return gradient

# Optimization function using gradient descent
def optimize(network, num_iterations=1000, learning_rate=0.01):
    input_size = len(network['Layers'][0]['weights'][0])
    input_data = [random.uniform(-1, 1) for _ in range(input_size)]
    best_score = float('inf')
    best_input = input_data[:]  # Initialize the best input to the starting input data

    for _ in range(num_iterations):
        gradient = compute_gradient(network, input_data)
        input_data = [input_data[i] - learning_rate * gradient[i] for i in range(len(input_data))]
        candidate_output = forward_pass(network, input_data)
        candidate_score = sum(abs(x) for x in candidate_output)
        if candidate_score < best_score:
            best_score = candidate_score
            best_input = input_data[:]  # Update the input data to the new candidate

    return best_input

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
                biases = eval(parts[1].strip())
                if isinstance(biases[0], list):
                    biases = [b[0] for b in biases]  # Flatten the list if it's a list of lists
                current_layer['biases'] = biases
            elif parts[0].startswith('Relu'):
                current_layer['relu'] = parts[1].strip().lower() == 'true'

    return networks

# Main function
def main():
    networks = parse_networks('networks.txt')
    with open('solutions.txt', 'w') as f_out:
        for i, network in enumerate(networks):
            try:
                input_data = optimize(network)  # Optimize the network
                input_string = ','.join(map(str, input_data))
                f_out.write(f"Network {i + 1}: {input_string}\n")
            except ValueError as e:
                print(f"Network {i + 1} encountered an error: {e}")

if __name__ == "__main__":
    main()
