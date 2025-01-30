#pragma once

#include "utils.h"
#include "activation_function.h"
#include "linear_function.h"

class Layer {
public:
    Layer(int start_dim, int end_dim, ActivationFunction activation_function);
    Vector Propagate(const Vector& x);
    Matrix BackPropagate(const Matrix& u, double learning_rate);
    int InputDim() const;
    int OutputDim() const;

private:
    Vector propagation_input_;
    LinearFunction linear_function_;
    ActivationFunction activation_function_;
};
