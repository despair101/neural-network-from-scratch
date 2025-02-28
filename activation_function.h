#pragma once

#include "linalg.h"

namespace NeuralNetworkFromScratch {

class ActivationFunction {
public:
    ActivationFunction(const std::function<double(double)>& function,
                       const std::function<double(double)>& derivative);
    Matrix Apply(const Matrix& X) const;
    Matrix JacobianCompose(const Matrix& U, const Matrix& A) const;

private:
    std::function<double(double)> function_;
    std::function<double(double)> derivative_;
};

ActivationFunction ReLU();

ActivationFunction Sigmoid();

ActivationFunction Id();

ActivationFunction SoftPlus();

ActivationFunction Sin();

}  // namespace NeuralNetworkFromScratch
