#pragma once
#include "linalg.h"

namespace NeuralNetworkFromScratch {

class LossFunction {
public:
    double Score(const Vector& y_true, const Vector& y_pred) const;
    Vector Gradient(const Vector& y_true, const Vector& y_pred) const;
};

}  // namespace NeuralNetworkFromScratch
