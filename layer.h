#pragma once

#include "activation_function.h"

namespace NNFS {

class Network;

class Layer {
    friend class Network;
    friend class SGD;
    friend class ADAM;

public:
    struct Data {
        Matrix cache_X_;
        Matrix cache_Y_;
        Matrix grad_A_;
        Vector grad_b_;
    };

    Layer(Index in_dim, Index out_dim, ActivationFunction activation_function);
    Layer(Matrix A, Vector b, ActivationFunction activation_function);
    Matrix Apply(const Matrix& X) const;
    Index InputDim() const;
    Index OutputDim() const;
    const Matrix& A();
    const Vector& b();
    void ShiftA(const Matrix& shift);
    void Shiftb(const Vector& shift);

private:
    const Matrix& AGrad();
    const Vector& bGrad();
    void SetZeroGrad();
    Matrix Propagate(Matrix&& X);
    Matrix BackPropagate(const Matrix& U);

    Matrix A_;
    Vector b_;
    ActivationFunction activation_function_;
    Data* data_;
};

}  // namespace NNFS
