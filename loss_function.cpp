#include "loss_function.h"

namespace NeuralNetworkFromScratch {

double LossFunction::Score(const Matrix& Y_true, const Matrix& Y_pred) const {
    return (Y_pred - Y_true).squaredNorm() / static_cast<double>(Y_true.cols());
}

Matrix LossFunction::Gradient(const Matrix& Y_true, const Matrix& Y_pred) const {
    return 2 * (Y_pred - Y_true) / static_cast<double>(Y_true.cols());
}

}  // namespace NeuralNetworkFromScratch
