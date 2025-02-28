#include "network.h"
#include <algorithm>

namespace NeuralNetworkFromScratch {

Network::Network(const std::vector<Layer>& layers) : layers_(layers) {
}

Network::Network(const std::vector<Index>& dimensions,
                 const std::vector<ActivationFunction>& activation_functions) {
    assert(dimensions.size() == activation_functions.size() + 1);
    assert(std::all_of(dimensions.begin(), dimensions.end(), [](int x) { return x > 0; }));
    layers_.reserve(activation_functions.size());
    for (int i = 0; i + 1 < dimensions.size(); ++i) {
        layers_.emplace_back(In{dimensions[i]}, Out{dimensions[i + 1]}, activation_functions[i]);
    }
}

void Network::Train(const Matrix& X, const Matrix& Y_true, int epochs, double learning_rate,
                    const LossFunction& loss_function) {
    assert(X.rows() == layers_.front().InputDim());
    assert(Y_true.rows() == layers_.back().OutputDim());
    for (int e = 0; e < epochs; ++e) {
        TrainEpoch(X, Y_true, learning_rate, loss_function);
    }
}

void Network::TrainEpoch(const Matrix& X, const Matrix& Y_true, double learning_rate,
                         const LossFunction& loss_function) {
    Matrix Y_pred = Propagate(X);
    Matrix u = loss_function.Gradient(Y_true, Y_pred).transpose();
    BackPropagate(u, learning_rate);
}

Matrix Network::Predict(const Matrix& X) const {
    assert(!layers_.empty());
    return Propagate(X);
}

Matrix Network::Propagate(Matrix X) {
    for (auto& layer : layers_) {
        X = layer.Propagate(X);
    }
    return X;
}

Matrix Network::Propagate(Matrix X) const {
    for (const auto& layer : layers_) {
        X = layer.Propagate(X);
    }
    return X;
}

void Network::BackPropagate(Matrix U, double learning_rate) {
    assert(!layers_.empty());
    for (auto layer_it = layers_.rbegin(); layer_it != layers_.rend(); ++layer_it) {
        U = layer_it->BackPropagate(U, learning_rate);
    }
}

}  // namespace NeuralNetworkFromScratch
