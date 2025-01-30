#pragma once

#include "utils.h"

class DifferentiableFunction {
public:
    DifferentiableFunction(const std::function<double(double)>& function,
                           const std::function<double(double)>& derivative);
    double Apply(double x) const;
    double Differentiate(double x) const;
    std::function<double(double)> Function() const;
    std::function<double(double)> Derivative() const;

private:
    std::function<double(double)> function_;
    std::function<double(double)> derivative_;
};

namespace Functions {

const DifferentiableFunction& ReLU();

const DifferentiableFunction& Sigmoid();

const DifferentiableFunction& Id();

const DifferentiableFunction& SoftPlus();

const DifferentiableFunction& Sin();

}  // namespace Functions