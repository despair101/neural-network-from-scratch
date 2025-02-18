#pragma once

#include "linalg.h"

namespace NeuralNetworkFromScratch {

class ActivationFunction {
public:
    ActivationFunction(const std::function<double(double)>& function,
                       const std::function<double(double)>& derivative);
    Vector Apply(const Vector& x) const;
    Matrix Jacobian(const Vector& x) const;

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
