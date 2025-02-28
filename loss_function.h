#pragma once
#include "linalg.h"

namespace NeuralNetworkFromScratch {

class LossFunction {
public:
    double Score(const Matrix& Y_true, const Matrix& Y_pred) const;
    Matrix Gradient(const Matrix& Y_true, const Matrix& Y_pred) const;
};

}  // namespace NeuralNetworkFromScratch
