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
        biases = layer['biases']
        layer_input = matmul([layer_input], weights)[0]
        layer_input = add_vectors(layer_input, biases)
        if layer['relu']:
            layer_input = [relu(x) for x in layer_input]
    return layer_input

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
                current_layer['biases'] = eval(parts[1].strip())[0]  # Flatten the biases list
            elif parts[0].startswith('Relu'):
                current_layer['relu'] = parts[1].strip().lower() == 'true'
    
    return networks

# Function to calculate the sum of absolute values
def sum_of_abs(values):
    return sum(abs(x) for x in values)

# Optimization function to minimize the output of the neural network
def optimize(network, num_iterations=10000, step_size=0.01):
    input_size = len(network['Layers'][0]['weights'])
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

# Main function to execute the program
def main():
    networks = parse_networks('networks.txt')
    with open('solutions.txt', 'w') as f_out:
        for i, network in enumerate(networks):
            print(f'Optimizing Network {i+1}')
            optimized_input = optimize(network)
            print('Optimized Input:', optimized_input)
            optimized_output = forward_pass(network, optimized_input)
            print('Network Output:', optimized_output)
            print('Sum of Absolute Values:', sum_of_abs(optimized_output))
            print('=' * 30)
            
            # Write the optimized input to the solutions.txt file
            f_out.write(','.join(map(str, optimized_input)) + '\n')

if __name__ == "__main__":
    main()
