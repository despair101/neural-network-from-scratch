#include "network.h"
#include <algorithm>

Network::Network(std::vector<Layer> layers, LossFunction loss_function)
    : layers_(std::move(layers)),
      loss_function_(loss_function),
      input_dim_(layers_.front().InputDim()),
      output_dim_(layers_.back().OutputDim()) {
}

Network::Network(const std::vector<int>& dimensions,
                 const std::vector<ActivationFunction>& activation_functions,
                 LossFunction loss_function) : loss_function_(loss_function) {
    assert(dimensions.size() == activation_functions.size() + 1);
    assert(std::all_of(dimensions.begin(), dimensions.end(), [](int x) {return x > 0; }));
    layers_.reserve(activation_functions.size());
    input_dim_ = dimensions.front();
    output_dim_ = dimensions.back();
    for (int i = 0; i + 1 < dimensions.size(); ++i) {
        layers_.emplace_back(dimensions[i], dimensions[i + 1], activation_functions[i]);
    }
}

void Network::Train(const Matrix& X, const Matrix& Y, int epochs, double learning_rate) {
    assert(X.cols() == input_dim_);
    assert(Y.cols() == output_dim_);
    for (int e = 0; e < epochs; ++e) {
        for (int i = 0; i < X.rows(); ++i) {
            Vector x = X.row(i).transpose();
            Vector y_pred = Propagate(x);
            Vector y_true = Y.row(i).transpose();
            Matrix u = loss_function_.Gradient(y_true, y_pred).transpose();
            BackPropagate(u, learning_rate);
        }
    }
}

Matrix Network::Predict(const Matrix& X) {
    Matrix Y(X.rows(), output_dim_);
    for (int i = 0; i < X.rows(); ++i) {
        Vector x = X.row(i).transpose();
        Y.row(i) = Propagate(x).transpose();
    }
    return Y;
}

Vector Network::Propagate(Vector x) {
    for (int i = 0; i < layers_.size(); ++i) {
        x = layers_[i].Propagate(x);
    }
    return x;
}

void Network::BackPropagate(Matrix y, double learning_rate) {
    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        y = layers_[i].BackPropagate(y, learning_rate);
    }
}

double Network::Score(const Matrix& Y_true, const Matrix& Y_pred) const {
    double loss = 0;
    int size = static_cast<int>(Y_true.rows());
    for (int i = 0; i < size; ++i) {
        loss += loss_function_.Score(Y_true.row(i).transpose(), Y_pred.row(i).transpose()) / size;
    }
    return loss;
}