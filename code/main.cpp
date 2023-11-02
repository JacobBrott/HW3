#include <iostream>
#include <vector>
#include "NeuralNetwork.h"

int main() {
    // Create a sample neural network for testing
    std::vector<std::vector<double>> weights1 = {{3.0}};
    std::vector<double> biases1 = {4.0};
    NetworkLayer layer1(weights1, biases1, false);
    
    NeuralNetwork nn;
    nn.addLayer(layer1);
    
    // Optimize the input for the neural network
    std::vector<double> input = {1.0};
    std::vector<double> output = nn.forward(input);
    
    // Print the output
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
