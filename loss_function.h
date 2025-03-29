#pragma once
#include "linalg.h"

namespace NNFS {

class LossFunction {
    using score_signature = double(const Matrix&, const Matrix&);
    using grad_signature = Matrix(const Matrix&, const Matrix&);

public:
    LossFunction(const std::function<score_signature>& score, const std::function<grad_signature>& gradient);
    double Score(const Matrix& Y_true, const Matrix& Y_pred) const;
    Matrix Gradient(const Matrix& Y_true, const Matrix& Y_pred) const;

private:
    std::function<score_signature> score_;
    std::function<grad_signature> gradient_;
};

LossFunction MSE();
LossFunction CrossEntropy();

}  // namespace NNFS
