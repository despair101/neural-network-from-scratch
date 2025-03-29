#include "activation_function.h"

namespace NNFS {

ActivationFunction::ActivationFunction(std::function<fun_signature> function,
                                       std::function<jacmul_signature> right_jacobian_multiplication)
    : function_(std::move(function)), right_jacobian_multiplication_(std::move(right_jacobian_multiplication)) {
}

Matrix ActivationFunction::Apply(const Matrix& X) const {
    assert(function_);
    Matrix res(X.rows(), X.cols());
    for (Index i = 0; i < X.cols(); ++i) {
        res.col(i) = function_(X.col(i));
    }
    return res;
}

Matrix ActivationFunction::JacobianCompose(const Matrix& U, const Matrix& X) const {
    assert(right_jacobian_multiplication_);
    Matrix res(U.rows(), X.rows());
    for (Index i = 0; i < U.rows(); ++i) {
        res.row(i) = right_jacobian_multiplication_(U.row(i), X.col(i));
    }
    return res;
}

ActivationFunction Id() {
    static ActivationFunction Id([](const Vector& x) -> Vector { return x; },
                                 [](const RowVector& u, const Vector& x) -> RowVector { return u; });
    return Id;
}

ActivationFunction ReLU() {
    static ActivationFunction ReLU([](const Vector& x) -> Vector { return x.array().max(0.0); },
                                   [](const RowVector& u, const Vector& x) -> RowVector {
                                       auto y = x.unaryExpr([](double t) -> double { return t > 0; }).array();
                                       return u.array() * y.transpose();
                                   });
    return ReLU;
}

ActivationFunction Sigmoid() {
    static ActivationFunction Sigmoid([](const Vector& x) -> Vector { return 1.0 / (1.0 + (-x.array()).exp()); },
                                      [](const RowVector& u, const Vector& x) -> RowVector {
                                          auto sigmoid = 1.0 / (1.0 + (-x.array()).exp());
                                          return u.array() * (sigmoid * (1 - sigmoid)).transpose();
                                      });
    return Sigmoid;
}

ActivationFunction Tanh() {
    static ActivationFunction Tanh([](const Vector& x) -> Vector { return 2 * Sigmoid().Apply(2 * x).array() - 1; },
                                   [](const RowVector& u, const Vector& x) -> RowVector {
                                       auto tanh = Sigmoid().Apply(2 * x).array() - 1;
                                       return u.array() * (1 - tanh.square()).transpose();
                                   });
    return Tanh;
}

ActivationFunction SoftMax() {
    static ActivationFunction SoftMax(
        [](const Vector& x) -> Vector {
            auto y = x.array() - x.maxCoeff();
            return y.exp() / y.exp().sum();
        },
        [](const RowVector& u, const Vector& x) -> RowVector {
            auto y = x.array() - x.maxCoeff();
            Vector sigma = y.exp() / y.exp().sum();
            return u * (Matrix(sigma.asDiagonal()) - sigma * sigma.transpose());
        });
    return SoftMax;
}

}  // namespace NNFS
