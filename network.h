#pragma once

#include <vector>
#include "linalg.h"
#include "layer.h"
#include "activation_function.h"
#include "loss_function.h"

namespace NeuralNetworkFromScratch {

class Network {
public:
    explicit Network(const std::vector<Layer>& layers);
    explicit Network(const std::vector<Index>& dimensions,
                     const std::vector<ActivationFunction>& activation_functions);
    void Train(const Matrix& X, const Matrix& Y_true, int epochs, double learning_rate,
               const LossFunction& loss_function);
    void TrainEpoch(const Matrix& X, const Matrix& Y_true, double learning_rate,
                    const LossFunction& loss_function);
    Matrix Predict(const Matrix& X) const;

private:
    Matrix Propagate(Matrix x);
    Matrix Propagate(Matrix x) const;
    void BackPropagate(Matrix y, double learning_rate);

    std::vector<Layer> layers_;
};

}  // namespace NeuralNetworkFromScratch
