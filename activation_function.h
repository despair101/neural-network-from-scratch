#pragma once

#include "linalg.h"

namespace NNFS {

class ActivationFunction {
    using fun_signature = Vector(const Vector&);
    using jacmul_signature = RowVector(const RowVector&, const Vector&);

public:
    ActivationFunction(std::function<fun_signature> function,
                       std::function<jacmul_signature> right_jacobian_multiplication);
    Matrix Apply(const Matrix& X) const;
    Matrix JacobianCompose(const Matrix& U, const Matrix& X) const;

private:
    std::function<fun_signature> function_;
    std::function<jacmul_signature> right_jacobian_multiplication_;
};

namespace Activation {

ActivationFunction Id();
ActivationFunction ReLU();
ActivationFunction Sigmoid();
ActivationFunction Tanh();
ActivationFunction SoftMax();

}  // namespace Activation

}  // namespace NNFS
