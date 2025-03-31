#include "layer.h"

namespace NNFS {

Layer::Layer(Index in_dim, Index out_dim, ActivationFunction activation_function)
    : A_(Matrix::Random(out_dim, in_dim)),
      b_(Vector::Random(out_dim)),
      activation_function_(std::move(activation_function)) {
}

Layer::Layer(Matrix A, Vector b, ActivationFunction activation_function)
    : A_(std::move(A)), b_(std::move(b)), activation_function_(std::move(activation_function)) {
    assert(A_.rows() == b_.rows());
}

Matrix Layer::Propagate(const Matrix& X) const {
    assert(A_.size() != 0 && b_.size() != 0);
    return activation_function_.Apply(A_ * X + b_.replicate(1, X.cols()));
}

Index Layer::InputDim() const {
    return A_.cols();
}

Index Layer::OutputDim() const {
    return A_.rows();
}

}  // namespace NNFS
