#pragma once

#include <Eigen/Core>

namespace NeuralNetworkFromScratch {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;
using Index = Eigen::Index;

enum In : Index;
enum Out : Index;

}  // namespace NeuralNetworkFromScratch
