#include "network.h"

namespace NNFS {

Network::Network(const std::vector<Layer>& layers) : layers_(layers) {
}

Network::Network(std::vector<Layer>&& layers) : layers_(std::move(layers)) {
}

Network::Network(const std::vector<Index>& sizes, const std::vector<ActivationFunction>& activation_functions) {
    assert(sizes.size() == activation_functions.size() + 1);
    assert(std::all_of(sizes.begin(), sizes.end(), [](Index x) { return x > 0; }));
    layers_.reserve(activation_functions.size());
    for (int i = 0; i + 1 < sizes.size(); ++i) {
        layers_.emplace_back(sizes[i], sizes[i + 1], activation_functions[i]);
    }
}

double Network::Score(const DataLoader& data_loader, const LossFunction& loss_function) const {
    double loss = 0;
    Index size = 0;
    for (auto [X, Y] : data_loader) {
        size += X.cols();
        loss += X.cols() * loss_function.Score(Y, Apply(X));
    }
    return loss / size;
}

Matrix Network::Predict(Matrix X) const {
    X = X.transpose().eval();
    for (const auto& layer : layers_) {
        X = layer.Apply(X);
    }
    return X.transpose();
}

void Network::InitData(DataKeeper& data_keeper) {
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i].data_ = &data_keeper.ptr_[i];
    }
}

void Network::SetNullData() {
    for (auto& layer : layers_) {
        layer.data_ = nullptr;
    }
}

Matrix Network::Propagate(Matrix&& X) {
    for (auto& layer : layers_) {
        X = layer.Propagate(std::move(X));
    }
    return X;
}

Matrix Network::Apply(Matrix X) const {
    for (auto& layer : layers_) {
        X = layer.Apply(X);
    }
    return X;
}

void Network::BackPropagate(Matrix U) {
    assert(!layers_.empty());
    for (size_t i = layers_.size(); i >= 1; --i) {
        U = layers_[i - 1].BackPropagate(U);
    }
}

const std::vector<Layer>& Network::Layers() const {
    return layers_;
}

}  // namespace NNFS
