#include "activation_function.h"

ActivationFunction::ActivationFunction(DifferentiableFunction function)
    : function_(std::move(function)) {
}

Vector ActivationFunction::Apply(const Vector& x) const {
    return x.unaryExpr(function_.Function());
}

Matrix ActivationFunction::Jacobian(const Vector& x) const {
    return x.unaryExpr(function_.Derivative()).asDiagonal();
}

DifferentiableFunction ActivationFunction::Function() const {
    return function_;
}
