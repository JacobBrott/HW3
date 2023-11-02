#pragma once
#include "NetworkLayer.h"
#include <vector>

class NeuralNetwork {
public:
    void addLayer(const NetworkLayer& layer);
    std::vector<double> forward(const std::vector<double>& input) const;

private:
    std::vector<NetworkLayer> layers;
};
