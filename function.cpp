#include "utils.h"
#include "function.h"

DifferentiableFunction::DifferentiableFunction(const std::function<double(double)>& function,
                                               const std::function<double(double)>& derivative)
    : function_(function), derivative_(derivative) {
}

double DifferentiableFunction::Apply(double x) const {
    return function_(x);
}

double DifferentiableFunction::Differentiate(double x) const {
    return derivative_(x);
}

std::function<double(double)> DifferentiableFunction::Function() const {
    return function_;
}

std::function<double(double)> DifferentiableFunction::Derivative() const {
    return derivative_;
}

namespace Functions {

const DifferentiableFunction& ReLU() {
    static auto f = [](double t) { return std::max(0.0, t); };
    static auto d = [](double t) { return t > 0 ? 1.0 : 0.0; };
    static DifferentiableFunction ReLU(f, d);
    return ReLU;
}

const DifferentiableFunction& Sigmoid() {
    static auto f = [](double t) { return 1.0 / (1.0 + std::exp(-t)); };
    static auto d = [](double t) { return f(t) * (1 - f(t)); };
    static DifferentiableFunction Sigmoid(f, d);
    return Sigmoid;
}

const DifferentiableFunction& Id() {
    static auto f = [](double t) { return t; };
    static auto d = [](double t) { return 1.0; };
    static DifferentiableFunction Id(f, d);
    return Id;
}

const DifferentiableFunction& SoftPlus() {
    static auto f = [](double t) { return std::log(1 + exp(t)); };
    static auto d = [](double t) { return Sigmoid().Apply(t); };
    static DifferentiableFunction SoftPlus(f, d);
    return SoftPlus;
}

const DifferentiableFunction& Sin() {
    static auto f = [](double t) { return std::sin(t); };
    static auto d = [](double t) { return std::cos(t); };
    static DifferentiableFunction Sin(f, d);
    return Sin;
}

}  // namespace Functions