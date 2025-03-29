#pragma once

#include "network.h"

namespace NNFS {

class SGD {
public:
    explicit SGD(double learning_rate);
    void Optimize(Network& network, const DataLoader& loader, const LossFunction& loss_function);

private:
    double learning_rate_;
};

}  // namespace NeuralNetworkFromScratch
