#include "activation_function.h"

namespace NeuralNetworkFromScratch {

ActivationFunction::ActivationFunction(const std::function<double(double)>& function,
                                       const std::function<double(double)>& derivative)
    : function_(function), derivative_(derivative) {
}

Matrix ActivationFunction::Apply(const Matrix& X) const {
    assert(function_);
    return X.unaryExpr(function_);
}

Matrix ActivationFunction::JacobianCompose(const Matrix& U, const Matrix& X) const {
    assert(derivative_);
    return U.array() * X.unaryExpr(derivative_).transpose().array();
}

ActivationFunction ReLU() {
    static ActivationFunction ReLU([](double t) { return std::max(0.0, t); },
                                   [](double t) { return t > 0 ? 1.0 : 0.0; });
    return ReLU;
}

ActivationFunction Sigmoid() {
    static ActivationFunction Sigmoid(
        [](double t) { return 1.0 / (1.0 + std::exp(-t)); },
        [](double t) { return 1.0 / (1.0 + std::exp(-t)) * (1 - 1.0 / (1.0 + std::exp(-t))); });
    return Sigmoid;
}

ActivationFunction Id() {
    static ActivationFunction Id([](double t) { return t; }, [](double t) { return 1.0; });
    return Id;
}

ActivationFunction SoftPlus() {
    static ActivationFunction SoftPlus([](double t) { return std::log(1 + exp(t)); },
                                       [](double t) { return 1.0 / (1.0 + std::exp(-t)); });
    return SoftPlus;
}

ActivationFunction Sin() {
    static ActivationFunction Sin([](double t) { return std::sin(t); },
                                  [](double t) { return std::cos(t); });
    return Sin;
}

}  // namespace NeuralNetworkFromScratch
