#pragma once
#include "utils.h"

class LinearFunction {
public:
    LinearFunction(int start_dim, int end_dim);
    Vector Apply(const Vector& x) const;
    Matrix GetA() const;
    Vector GetB() const;
    void Shift(const Matrix& delta_A, const Matrix& delta_b);
    int InputDim() const;
    int OutputDim() const;

private:
    Matrix A;
    Vector b;
};
