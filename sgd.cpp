#include "sgd.h"

namespace NNFS {

SGD::SGD(double learning_rate) : learning_rate_(learning_rate) {
}

void SGD::Optimize(Network& network, const DataLoader& loader, const LossFunction& loss_function) {
    for (auto [X, Y] : loader) {
        network.BackPropagate(loss_function.Gradient(Y, network.Propagate(std::move(X))).transpose());
        for (auto& layer : network.layers_) {
            layer.A_ -= learning_rate_ * layer.data_->dA_;
            layer.b_ -= learning_rate_ * layer.data_->db_;
            layer.data_->dA_.setZero();
            layer.data_->db_.setZero();
        }
    }
}

}  // namespace NNFS
