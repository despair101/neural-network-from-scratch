#pragma once

#include "activation_function.h"

namespace NNFS {

class Layer {
public:
    Layer(Index in_dim, Index out_dim, ActivationFunction activation_function);
    Matrix Propagate(const Matrix& X) const;
    Index InputDim() const;
    Index OutputDim() const;

    Matrix A_;
    Vector b_;
    ActivationFunction activation_function_;
};

}  // namespace NNFS
