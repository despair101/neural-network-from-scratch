#include "layer.h"

namespace NeuralNetworkFromScratch {

Layer::Layer(In in_dim, Out out_dim, const ActivationFunction& activation_function)
    : A_(Matrix::Random(out_dim, in_dim)),
      b_(Vector::Random(out_dim)),
      activation_function_(activation_function) {
}

Matrix Layer::Propagate(const Matrix& X) {
    assert(A_.size() != 0 && b_.size() != 0);
    cache_ = {X, A_ * X + b_.replicate(1, X.cols())};
    return activation_function_.Apply(cache_.Y);
}

Matrix Layer::Propagate(const Matrix& X) const {
    assert(A_.size() != 0 && b_.size() != 0);
    return activation_function_.Apply(A_ * X + b_.replicate(1, X.cols()));
}

Matrix Layer::BackPropagate(const Matrix& U, double learning_rate) {
    assert(A_.size() != 0 && b_.size() != 0);
    Matrix U_jac = activation_function_.JacobianCompose(U, cache_.Y);
    Matrix U_prop = U_jac * A_;
    A_ -= learning_rate * (cache_.X * U_jac).transpose();
    b_ -= learning_rate * U_jac.transpose().rowwise().sum();
    return U_prop;
}

Index Layer::InputDim() const {
    return A_.cols();
}

Index Layer::OutputDim() const {
    return A_.rows();
}

}  // namespace NeuralNetworkFromScratch
