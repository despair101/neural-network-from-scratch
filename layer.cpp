#include "layer.h"

Layer::Layer(int start_dim, int end_dim, ActivationFunction activ)
    : linear_function_(start_dim, end_dim), activation_function_(std::move(activ)) {
}

Vector Layer::Propagate(const Vector& x) {
    propagation_input_ = x;
    return activation_function_.Apply(linear_function_.Apply(x));
}

Matrix Layer::BackPropagate(const Matrix& u, double learning_rate) {
    const Matrix& jacobian =
        activation_function_.Jacobian(linear_function_.Apply(propagation_input_));
    Matrix u_prop = u * jacobian * linear_function_.GetA();
    linear_function_.Shift(-learning_rate * (propagation_input_ * (u * jacobian)).transpose(),
                           -learning_rate * (u * jacobian).transpose());
    return u_prop;
}

int Layer::InputDim() const {
    return linear_function_.InputDim();
}

int Layer::OutputDim() const {
    return linear_function_.OutputDim();
}
