#include "network.h"

namespace NNFS {

Network::Network(const std::vector<Layer>& layers) {
    layers_.reserve(layers.size());
    for (const auto& layer : layers) {
        layers_.emplace_back(layer);
    }
}

Network::Network(const std::vector<Index>& dimensions, const std::vector<ActivationFunction>& activation_functions) {
    assert(dimensions.size() == activation_functions.size() + 1);
    assert(std::all_of(dimensions.begin(), dimensions.end(), [](Index x) { return x > 0; }));
    layers_.reserve(activation_functions.size());
    for (int i = 0; i + 1 < dimensions.size(); ++i) {
        layers_.emplace_back(dimensions[i], dimensions[i + 1], activation_functions[i]);
    }
}

double Network::Score(const DataLoader& loader, const LossFunction& loss_function) const {
    double loss = 0;
    Index size = 0;
    for (auto [X, Y] : loader) {
        size += X.cols();
        loss += X.cols() * loss_function.Score(Y, Propagate(X));
    }
    return loss / size;
}

Matrix Network::Predict(const Matrix& X) const {
    assert(!layers_.empty());
    return Propagate(X.transpose()).transpose();
}

Matrix Network::Propagate(Matrix&& X) {
    for (auto& layer : layers_) {
        X = layer.Propagate(std::move(X));
    }
    return X;
}

Matrix Network::Propagate(Matrix X) const {
    for (const auto& layer : layers_) {
        X = layer.Propagate(X);
    }
    return X;
}

void Network::BackPropagate(Matrix U) {
    assert(!layers_.empty());
    for (size_t i = layers_.size(); i >= 1; --i) {
        U = layers_[i - 1].BackPropagate(U);
    }
}

Network::ExtendedLayer::ExtendedLayer(const Layer& layer) : Layer(layer) {
}

Network::ExtendedLayer::ExtendedLayer(Index in_dim, Index out_dim, const ActivationFunction& activation_function)
    : Layer(in_dim, out_dim, activation_function) {
}

void Network::ExtendedLayer::InitData() {
    data_ = std::make_unique<Data>();
}

void Network::ExtendedLayer::ReleaseData() {
    data_.reset();
}

Matrix Network::ExtendedLayer::Propagate(const Matrix& X) const {
    return Layer::Propagate(X);
}

Matrix Network::ExtendedLayer::Propagate(Matrix&& X) {
    assert(A_.size() != 0 && b_.size() != 0);
    data_->X_cache_ = std::move(X);
    data_->Y_cache_ = A_ * data_->X_cache_ + b_.replicate(1, data_->X_cache_.cols());
    return activation_function_.Apply(data_->Y_cache_);
}

Matrix Network::ExtendedLayer::BackPropagate(const Matrix& U) {
    assert(A_.size() != 0 && b_.size() != 0);
    Matrix U_jac = activation_function_.JacobianCompose(U, data_->Y_cache_);
    data_->dA_ = (data_->X_cache_ * U_jac).transpose();
    data_->db_ = U_jac.transpose().rowwise().sum();
    return U_jac * A_;
}

std::vector<Layer> Network::Layers() const {
    std::vector<Layer> layers;
    layers.reserve(layers_.size());
    for (auto& layer : layers_) {
        layers.push_back(layer);
    }
    return layers;
}

}  // namespace NNFS
