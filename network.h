#pragma once

#include "linalg.h"
#include "layer.h"
#include "activation_function.h"
#include "loss_function.h"
#include "data_loader.h"

#include <vector>
#include <iostream>

namespace NNFS {

class SGD;
class ADAM;

class Network {
public:
    explicit Network(const std::vector<Layer>& layers);
    explicit Network(const std::vector<Index>& sizes, const std::vector<ActivationFunction>& activation_functions);
    template <typename Optimizer>
        requires(std::is_same_v<Optimizer, SGD> || std::is_same_v<Optimizer, ADAM>)
    void Train(DataLoader& data_loader, Optimizer& optimizer, const LossFunction& loss_function, size_t epochs,
               bool debug = false, bool shuffle = true) {
        for (auto& layer : layers_) {
            layer.InitData();
        }
        for (size_t e = 0; e < epochs; ++e) {
            if (shuffle) {
                data_loader.Shuffle();
            }
            optimizer.Optimize(*this, data_loader, loss_function);
            if (debug) {
                std::cerr << "Epoch: " << e << ", Loss: " << Score(data_loader, loss_function) << std::endl;
            }
        }
        for (auto& layer : layers_) {
            layer.ReleaseData();
        }
    }
    double Score(const DataLoader& data_loader, const LossFunction& loss_function) const;
    Matrix Predict(const Matrix& X) const;
    std::vector<Layer> Layers() const;

private:
    class ExtendedLayer : public Layer {
    public:
        struct Data {
            Matrix cache_X_;
            Matrix cache_Y_;
            Matrix grad_A_;
            Vector grad_b_;
        };
        Matrix Propagate(Matrix&& X);
        Matrix Propagate(const Matrix& X) const;
        Matrix BackPropagate(const Matrix& U);
        void InitData();
        void ReleaseData();

        std::unique_ptr<Data> data_;
        explicit ExtendedLayer(const Layer& layer);
        ExtendedLayer(Index in_dim, Index out_dim, const ActivationFunction& activation_function);
    };

    Matrix Propagate(Matrix&& X);
    Matrix Propagate(Matrix X) const;
    void BackPropagate(Matrix Y);

    std::vector<ExtendedLayer> layers_;

    friend class SGD;
    friend class ADAM;
};

}  // namespace NNFS
