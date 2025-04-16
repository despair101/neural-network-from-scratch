#include "layer.h"

namespace NNFS {

Layer::Layer(Index in_dim, Index out_dim, ActivationFunction activation_function)
    : A_(Matrix::Random(out_dim, in_dim)),
      b_(Vector::Random(out_dim)),
      activation_function_(std::move(activation_function)),
      data_(nullptr) {
}

Layer::Layer(Matrix A, Vector b, ActivationFunction activation_function)
    : A_(std::move(A)), b_(std::move(b)), activation_function_(std::move(activation_function)), data_(nullptr) {
    assert(A_.rows() == b_.rows());
}

Matrix Layer::Apply(const Matrix& X) const {
    assert(A_.size() != 0 && b_.size() != 0);
    return activation_function_.Apply((A_ * X).colwise() + b_);
}

Matrix Layer::Propagate(Matrix&& X) {
    assert(A_.size() != 0 && b_.size() != 0);
    assert(data_);
    data_->cache_X_ = std::move(X);
    data_->cache_Y_ = (A_ * data_->cache_X_).colwise() + b_;
    return activation_function_.Apply(data_->cache_Y_);
}

Matrix Layer::BackPropagate(const Matrix& U) {
    assert(A_.size() != 0 && b_.size() != 0);
    assert(data_);
    Matrix U_jac = activation_function_.JacobianCompose(U, data_->cache_Y_);
    data_->grad_A_ = (data_->cache_X_ * U_jac).transpose();
    data_->grad_b_ = U_jac.transpose().rowwise().sum();
    return U_jac * A_;
}

const Matrix& Layer::A() {
    return A_;
}

const Vector& Layer::b() {
    return b_;
}
const Matrix& Layer::AGrad() {
    assert(data_);
    return data_->grad_A_;
}

const Vector& Layer::bGrad() {
    assert(data_);
    return data_->grad_b_;
}

void Layer::ShiftA(const Matrix& shift) {
    assert(A_.rows() == shift.rows() && A_.cols() == shift.cols());
    A_ += shift;
}

void Layer::Shiftb(const Vector& shift) {
    assert(b_.rows() == shift.rows() && b_.cols() == shift.cols());
    b_ += shift;
}

void Layer::SetZeroGrad() {
    assert(data_);
    data_->grad_A_.setZero();
    data_->grad_b_.setZero();
}

Index Layer::InputDim() const {
    return A_.cols();
}

Index Layer::OutputDim() const {
    return A_.rows();
}

}  // namespace NNFS
