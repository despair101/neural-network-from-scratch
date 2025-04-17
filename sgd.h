#pragma once

#include "network.h"

namespace NNFS {

class SGD {
    friend class Network;

public:
    explicit SGD(double learning_rate);

private:
    void Optimize(Network& network, const DataLoader& data_loader, const LossFunction& loss_function);

    double learning_rate_;
};

}  // namespace NNFS
