#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

class NeuralNetwork {
public:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> activations;

    // Define a constructor if necessary

    // Define a method to perform matrix multiplication
    std::vector<double> matMul(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) {
        std::vector<double> result(mat.size(), 0.0);
        for (size_t i = 0; i < mat.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += mat[i][j] * vec[j];
            }
        }
        return result;
    }

    // Define the activation function (e.g., sigmoid, relu)
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Compute the forward pass
    std::vector<double> forward(const std::vector<double>& input) {
        activations = matMul(weights, input);
        for (size_t i = 0; i < activations.size(); ++i) {
            activations[i] = sigmoid(activations[i] + biases[i]);
        }
        return activations;
    }

    // Add any other necessary methods
};
