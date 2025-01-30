#include "loss_function.h"

double LossFunction::Score(const Vector& y_true, const Vector& y_pred) const {
    return (y_true - y_pred).squaredNorm();
}

Vector LossFunction::Gradient(const Vector& y_true, const Vector& y_pred) const {
    return 2 * (y_pred - y_true);
}