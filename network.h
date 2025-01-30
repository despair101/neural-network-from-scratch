#pragma once

#include <vector>
#include "utils.h"
#include "layer.h"
#include "activation_function.h"
#include "loss_function.h"

class Network {
public:
    Network(std::vector<Layer> layers, LossFunction loss_function);
    Network(const std::vector<int>& dimensions,
            const std::vector<ActivationFunction>& activation_functions,
            LossFunction loss_function);
    void Train(const Matrix& X, const Matrix& Y, int epochs = 1000, double learning_rate = 0.01);
    Matrix Predict(const Matrix& X);
    double Score(const Matrix& Y_true, const Matrix& Y_pred) const;

private:
    Vector Propagate(Vector x);
    void BackPropagate(Matrix y, double learning_rate);

    std::vector<Layer> layers_;
    LossFunction loss_function_;
    int input_dim_;
    int output_dim_;
};
