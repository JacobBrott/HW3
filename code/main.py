import random
import math

def relu(x):
    return max(0, x)

def network_forward_pass(network, inputs):
    layer_outputs = inputs
    for layer in network:
        weights, bias = layer['weights'], layer['bias']
        next_layer_outputs = []
        for w, b in zip(weights, bias):
            next_layer_outputs.append(relu(sum(i * wi for i, wi in zip(layer_outputs, w)) + b))
        layer_outputs = next_layer_outputs
    return layer_outputs

def simulated_annealing(network, input_size, temperature=1.0, cooling_rate=0.99, num_iterations=1000):
    current_solution = [random.uniform(-10, 10) for _ in range(input_size)]
    current_energy = sum(network_forward_pass(network, current_solution))

    best_solution = current_solution
    best_energy = current_energy

    for _ in range(num_iterations):
        new_solution = [i + random.uniform(-1, 1) for i in current_solution]
        new_energy = sum(network_forward_pass(network, new_solution))

        if new_energy < best_energy:
            best_solution, best_energy = new_solution, new_energy

        if new_energy < current_energy or random.uniform(0, 1) < math.exp((current_energy - new_energy) / temperature):
            current_solution, current_energy = new_solution, new_energy

        temperature *= cooling_rate

    return best_solution

def parse_networks(file_name):
    networks = []
    network = {}
    layer = None
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            key, value = line.split(':')
            key, value = key.strip(), value.strip()
            if key == 'Layers':
                if network:
                    networks.append(network)
                    network = {}
            elif key.startswith('Rows') or key.startswith('Cols') or key.startswith('Relu'):
                layer_key = key.split(' ')[0] + 's'
                layer_index = int(key.split(' ')[1]) - 1
                if layer_key not in network:
                    network[layer_key] = {}
                network[layer_key][layer_index] = eval(value)
            elif key.startswith('Weights') or key.startswith('Biases'):
                layer_key = key.split(' ')[0] + 's'
                layer_index = int(key.split(' ')[1]) - 1
                if layer_key not in network:
                    network[layer_key] = {}
                network[layer_key][layer_index] = eval(value)
            elif key == 'Example_Input' or key == 'Example_Output':
                network[key] = eval(value)
            else:
                network[key] = eval(value)
    if network:
        networks.append(network)
    return networks

if __name__ == "__main__":
    networks = parse_networks('networks.txt')
    print(networks)
    with open('solutions.txt', 'w') as file:
        for network in networks:
            input_size = len(network[0]['weights'][0])  # Assuming all layers have the same number of inputs
            minimizing_input = simulated_annealing(network, input_size)
            file.write(','.join(map(str, minimizing_input)) + '\n')
