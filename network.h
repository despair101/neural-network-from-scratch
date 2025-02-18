#pragma once

#include <vector>
#include "linalg.h"
#include "layer.h"
#include "activation_function.h"
#include "loss_function.h"

namespace NeuralNetworkFromScratch {

class Network {
public:
    Network(const std::vector<Layer>& layers, LossFunction loss_function);
    Network(const std::vector<int>& dimensions,
            const std::vector<ActivationFunction>& activation_functions,
            LossFunction loss_function);
    void Train(const Matrix& X, const Matrix& Y, int epochs, double learning_rate);
    Matrix Predict(const Matrix& X);
    double Score(const Matrix& Y_true, const Matrix& Y_pred) const;

private:
    Vector Propagate(Vector x);
    void BackPropagate(Matrix y, double learning_rate);

    std::vector<Layer> layers_;
    LossFunction loss_function_;
};

}  // namespace NeuralNetworkFromScratch
