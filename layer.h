#pragma once

#include "activation_function.h"

namespace NeuralNetworkFromScratch {

struct LayerCache {
    Vector x;
    Vector y;
};

class Layer {
public:
    Layer(int start_dim, int end_dim, ActivationFunction activation_function);
    Vector Propagate(const Vector& x);
    Matrix BackPropagate(const Matrix& u, double learning_rate);
    Index InputDim() const;
    Index OutputDim() const;

private:
    LayerCache cache_;
    Matrix A_;
    Vector b_;
    ActivationFunction activation_function_;
};

}  // namespace NeuralNetworkFromScratch
