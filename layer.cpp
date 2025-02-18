#include "layer.h"

namespace NeuralNetworkFromScratch {

Layer::Layer(int start_dim, int end_dim, ActivationFunction activation_function)
    : A_(Matrix::Random(end_dim, start_dim)),
      b_(Vector::Random(end_dim)),
      activation_function_(std::move(activation_function)) {
}

Vector Layer::Propagate(const Vector& x) {
    assert(A_.size() != 0 && b_.size() != 0);
    cache_ = {x, A_ * x + b_};
    return activation_function_.Apply(cache_.y);
}

Matrix Layer::BackPropagate(const Matrix& u, double learning_rate) {
    assert(A_.size() != 0 && b_.size() != 0);
    Matrix jacobian = activation_function_.Jacobian(cache_.y);
    Matrix u_prop = u * jacobian * A_;
    A_ -= learning_rate * (cache_.x * (u * jacobian)).transpose();
    b_ -= learning_rate * (u * jacobian).transpose();
    return u_prop;
}

Index Layer::InputDim() const {
    return A_.cols();
}

Index Layer::OutputDim() const {
    return A_.rows();
}

}  // namespace NeuralNetworkFromScratch
