#pragma once

#include "network.h"

namespace NNFS {

class ADAM {
public:
    ADAM(double learning_rate, double moment1_coef, double moment2_coef, double eps);
    void Optimize(Network& network, const DataLoader& loader, const LossFunction& loss_function);

private:
    Matrix NextM(const Matrix& m, const Matrix& g) const;
    Matrix NextV(const Matrix& v, const Matrix& g) const;
    Matrix NextW(const Matrix& w, const Matrix& m, const Matrix& v) const;

    std::vector<std::pair<Matrix, Vector>> m_;
    std::vector<std::pair<Matrix, Vector>> v_;
    double learning_rate_;
    double m_coef_;
    double v_coef_;
    double m_coef_pw_;
    double v_coef_pw_;
    double eps_;
};

}  // namespace NNFS
