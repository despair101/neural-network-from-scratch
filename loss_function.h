#pragma once
#include "utils.h"

class LossFunction {
public:
    double Score(const Vector& y_true, const Vector& y_pred) const;
    Vector Gradient(const Vector& y_true, const Vector& y_pred) const;
};
