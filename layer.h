#pragma once

#include "activation_function.h"

namespace NeuralNetworkFromScratch {

struct LayerCache {
    Matrix X;
    Matrix Y;
};

class Layer {
public:
    Layer(In in_dim, Out out_dim, const ActivationFunction& activation_function);
    Matrix Propagate(const Matrix& X);
    Matrix Propagate(const Matrix& X) const;
    Matrix BackPropagate(const Matrix& U, double learning_rate);
    Index InputDim() const;
    Index OutputDim() const;

private:
    LayerCache cache_;
    Matrix A_;
    Vector b_;
    ActivationFunction activation_function_;
};

}  // namespace NeuralNetworkFromScratch
