#pragma once
#include <vector>

class NetworkLayer {
public:
    NetworkLayer(const std::vector<std::vector<double>>& weights, const std::vector<double>& biases, bool isRelu);
    std::vector<double> forward(const std::vector<double>& input) const;

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    bool isRelu;
    static std::vector<double> matMul(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec);
    static double relu(double x);
};
