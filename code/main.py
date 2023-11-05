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
    with open(filename, 'r') as f:
        network_data = f.read().strip().split('\n\n')
        for net in network_data:
            lines = net.split('\n')
            layer_count = int(lines[0].split(':')[1].strip())
            layers = []
            idx = 1
            for _ in range(layer_count):
                weights = eval(lines[idx+2].split(':')[1].strip())
                biases = eval(lines[idx+3].split(':')[1].strip())
                relu_activation = lines[idx+4].split(':')[1].strip().lower() == 'true'
                layers.append({'weights': weights, 'biases': [b[0] for b in biases], 'relu': relu_activation})
                idx += 5
            networks.append({'Layers': layers})
    return networks

# Main function
def main():
    networks = parse_networks('networks.txt')
    with open('solutions.txt', 'w') as f_out:
        for i, network in enumerate(networks):
            input_data = optimize(network)  # Optimize the network
            input_string = ','.join(map(str, input_data))
            f_out.write(f"{input_string}\n")

if __name__ == "__main__":
    main()
