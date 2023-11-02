#include "NetworkLayer.h"
#include <cmath>
#include <algorithm>

NetworkLayer::NetworkLayer(const std::vector<std::vector<double>>& weights, const std::vector<double>& biases, bool isRelu)
    : weights(weights), biases(biases), isRelu(isRelu) {}

std::vector<double> NetworkLayer::forward(const std::vector<double>& input) const {
    // Multiply input by weights and add biases
    std::vector<double> activations = matMul(weights, input);
    for (size_t i = 0; i < activations.size(); ++i) {
        // Apply ReLU activation if needed
        activations[i] = activations[i] + biases[i];
        if (isRelu) {
            activations[i] = relu(activations[i]);
        }
    }
    return activations;
}

std::vector<double> NetworkLayer::matMul(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) {
    std::vector<double> result(mat.size(), 0.0);
    for (size_t i = 0; i < mat.size(); ++i) {
        for (size_t j = 0; j < vec.size(); ++j) {
            result[i] += mat[i][j] * vec[j];
        }
    }
    return result;
}

double NetworkLayer::relu(double x) {
    return std::max(0.0, x);
}
