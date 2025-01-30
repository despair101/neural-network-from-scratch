#include "linear_function.h"
#include <iostream>

LinearFunction::LinearFunction(int start_dim, int end_dim)
    : A(Matrix::Random(end_dim, start_dim)), b(Vector::Random(end_dim)) {
}

Vector LinearFunction::Apply(const Vector& x) const {
    assert(x.rows() == InputDim());
    return A * x + b;
}

Matrix LinearFunction::GetA() const {
    return A;
}

Vector LinearFunction::GetB() const {
    return b;
}

void LinearFunction::Shift(const Matrix& delta_A, const Matrix& delta_b) {
    A += delta_A;
    b += delta_b;
}

int LinearFunction::InputDim() const {
    return static_cast<int>(A.cols());
}

int LinearFunction::OutputDim() const {
    return static_cast<int>(A.rows());
}
