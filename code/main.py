import random

# Activation function: ReLU
def relu(x):
    return max(0, x)

# Matrix multiplication
def matmul(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum = 0
            for k in range(len(B)):
                sum += A[i][k] * B[k][j]
            row.append(sum)
        result.append(row)
    return result

# Addition of two vectors
def add_vectors(a, b):
    return [x + y for x, y in zip(a, b)]

# Apply activation function element-wise
def apply_activation(func, matrix):
    return [[func(x) for x in row] for row in matrix]

# Forward pass through the neural network
def forward_pass(network, input_data):
    layer_input = input_data
    for layer in network['Layers']:
        weights = layer['weights']
        biases = layer['biases'][0]  # Adjusting this line to access the first row of biases matrix
        layer_input = matmul([layer_input], weights)[0]
        layer_input = add_vectors(layer_input, biases)
        if layer['relu']:
            layer_input = apply_activation(relu, [layer_input])[0]  # Adjusting this line to treat layer_input as a matrix
    return layer_input

# Parse networks from file
def parse_networks(filename):
    networks = []
    current_network = None
    current_layer = None
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            key = parts[0].strip()
            value = ':'.join(parts[1:]).strip()
            
            if key == 'Layers':
                if current_network is not None:
                    networks.append(current_network)
                current_network = {'Layers': []}
            elif key.startswith('Rows') or key.startswith('Cols'):
                pass  # We can infer Rows and Cols from Weights matrix
            elif key.startswith('Weights') or key.startswith('Biases'):
                matrix = eval(value)
                current_layer[key.lower()] = matrix
            elif key == 'Relu':
                current_layer['relu'] = value.lower() == 'true'
            elif key.startswith('Example_Input') or key.startswith('Example_Output'):
                pass  # These are not used for network definition
            else:
                current_layer = {}
                current_network['Layers'].append(current_layer)
                
        networks.append(current_network)  # Add the last network
    return networks

# Function to calculate the sum of absolute values
def sum_of_abs(values):
    return sum(abs(x) for x in values)

# Optimization function to minimize the output of the neural network
def optimize(network, num_iterations=10000, learning_rate=0.01):
    input_size = len(network['Layers'][0]['weights'][0])  # Adjusting this line to access the first row of weights matrix
    best_input = [random.uniform(-1, 1) for _ in range(input_size)]
    best_output = forward_pass(network, best_input)
    best_score = sum_of_abs(best_output)
    
    for _ in range(num_iterations):
        candidate_input = [x + random.uniform(-learning_rate, learning_rate) for x in best_input]
        candidate_output = forward_pass(network, candidate_input)
        candidate_score = sum_of_abs(candidate_output)
        if candidate_score < best_score:
            best_input, best_output, best_score = candidate_input, candidate_output, candidate_score
    
    return best_input

# Main function to execute the program
def main():
    networks = parse_networks('networks.txt')
    for i, network in enumerate(networks, 1):
        print(f'Optimizing Network {i}')
        optimized_input = optimize(network)
        print('Optimized Input:', optimized_input)
        optimized_output = forward_pass(network, optimized_input)
        print('Network Output:', optimized_output)
        print('Sum of Absolute Values:', sum_of_abs(optimized_output))
        print('=' * 30)

if __name__ == "__main__":
    main()
