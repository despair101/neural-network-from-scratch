#include "sgd.h"

namespace NNFS {

SGD::SGD(double learning_rate) : learning_rate_(learning_rate) {
}

void SGD::Optimize(Network& network, const DataLoader& data_loader, const LossFunction& loss_function) {
    for (auto [X, Y] : data_loader) {
        network.BackPropagate(loss_function.Gradient(Y, network.Propagate(std::move(X))).transpose());
        for (auto& layer : network.layers_) {
            layer.ShiftA(-learning_rate_ * layer.AGrad());
            layer.Shiftb(-learning_rate_ * layer.bGrad());
            layer.SetZeroGrad();
        }
    }
}

}  // namespace NNFS
