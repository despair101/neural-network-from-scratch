#pragma once

#include "utils.h"
#include "function.h"

class ActivationFunction {
public:
    ActivationFunction(DifferentiableFunction function);
    Vector Apply(const Vector& x) const;
    Matrix Jacobian(const Vector& x) const;
    DifferentiableFunction Function() const;

private:
    DifferentiableFunction function_;
};
