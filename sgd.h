#pragma once

#include "network.h"

namespace NNFS {

class SGD {
public:
    explicit SGD(double learning_rate);

private:
    void Optimize(Network& network, const DataLoader& data_loader, const LossFunction& loss_function);

    double learning_rate_;

    friend class Network;
};

}  // namespace NNFS
