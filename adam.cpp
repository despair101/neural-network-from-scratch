#include "adam.h"

namespace NNFS {

ADAM::ADAM(double learning_rate, double m_coef, double v_coef, double eps)
    : learning_rate_(learning_rate),
      m_coef_(m_coef),
      v_coef_(v_coef),
      m_coef_pw_(m_coef),
      v_coef_pw_(v_coef),
      eps_(eps) {
}

void ADAM::OptimizeInit(const std::vector<Layer>& layers) {
    if (m_.empty()) {
        m_.resize(layers.size());
        v_.resize(layers.size());
        for (size_t i = 0; i < layers.size(); ++i) {
            m_[i] = {Matrix::Zero(layers[i].OutputDim(), layers[i].InputDim()), Vector::Zero(layers[i].OutputDim())};
            v_[i] = m_[i];
        }
    }
}

void ADAM::Optimize(Network& network, const DataLoader& data_loader, const LossFunction& loss_function) {
    auto& layers = network.layers_;
    OptimizeInit(layers);
    for (auto [X, Y] : data_loader) {
        network.BackPropagate(loss_function.Gradient(Y, network.Propagate(std::move(X))).transpose());
        for (size_t i = 0; i < layers.size(); ++i) {
            auto& layer = layers[i];
            auto& [mA, mb] = m_[i];
            auto& [vA, vb] = v_[i];
            auto& grad_A = layer.AGrad();
            auto& grad_b = layer.bGrad();
            mA = NextM(mA, grad_A), mb = NextM(mb, grad_b);
            vA = NextV(vA, grad_A), vb = NextV(vb, grad_b);
            layer.ShiftA(GetWShift(layer.A(), mA, vA));
            layer.Shiftb(GetWShift(layer.b(), mb, vb));
            layer.SetZeroGrad();
        }
        m_coef_pw_ *= m_coef_;
        v_coef_pw_ *= v_coef_;
    }
}

Matrix ADAM::NextM(const Matrix& m, const Matrix& g) const {
    return m_coef_ * m + (1 - m_coef_) * g;
}

Matrix ADAM::NextV(const Matrix& v, const Matrix& g) const {
    return v_coef_ * v + (1 - v_coef_) * Matrix(g.array().square());
}

Matrix ADAM::GetWShift(const Matrix& w, const Matrix& m, const Matrix& v) const {
    return -learning_rate_ * Matrix((m.array() / (1 - m_coef_pw_)) / ((v.array() / (1 - v_coef_pw_)).sqrt() + eps_));
}

}  // namespace NNFS
