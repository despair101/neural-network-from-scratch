#include "loss_function.h"
#include "activation_function.h"

namespace NNFS {

LossFunction::LossFunction(const std::function<score_signature>& score, const std::function<grad_signature>& gradient)
    : score_(score), gradient_(gradient) {
}

double LossFunction::Score(const Matrix& Y_true, const Matrix& Y_pred) const {
    assert(Y_true.rows() == Y_pred.rows() && Y_true.cols() == Y_pred.cols());
    return score_(Y_true, Y_pred);
}

Matrix LossFunction::Gradient(const Matrix& Y_true, const Matrix& Y_pred) const {
    assert(Y_true.rows() == Y_pred.rows() && Y_true.cols() == Y_pred.cols());
    return gradient_(Y_true, Y_pred);
}

LossFunction MSE() {
    static LossFunction MSE(
        [](const Matrix& Y_true, const Matrix& Y_pred) -> double {
            return (Y_pred - Y_true).squaredNorm() / static_cast<double>(Y_true.cols());
        },
        [](const Matrix& Y_true, const Matrix& Y_pred) -> Matrix {
            return 2 * (Y_pred - Y_true) / static_cast<double>(Y_true.cols());
        });
    return MSE;
}
//
LossFunction CrossEntropy() {
    static LossFunction CrossEntropy(
        [](const Matrix& Y_true, const Matrix& Y_pred) -> double {
            return -(Y_true.array() * Y_pred.array().log()).sum() / static_cast<double>(Y_true.rows());
        },
        [](const Matrix& Y_true, const Matrix& Y_pred) -> Matrix {
            return -(Y_true.array() / Y_pred.array()) / static_cast<double>(Y_true.rows());
        });
    return CrossEntropy;
}

}  // namespace NNFS
