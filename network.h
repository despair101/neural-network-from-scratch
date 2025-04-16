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
    friend class SGD;
    friend class ADAM;

public:
    explicit Network(const std::vector<Layer>& layers);
    explicit Network(std::vector<Layer>&& layers);
    explicit Network(const std::vector<Index>& sizes, const std::vector<ActivationFunction>& activation_functions);
    template <typename Optimizer>
        requires(std::is_same_v<Optimizer, SGD> || std::is_same_v<Optimizer, ADAM>)
    void Train(DataLoader& data_loader, Optimizer& optimizer, const LossFunction& loss_function, size_t epochs,
               bool debug = false, bool shuffle = true) {
        DataKeeper keeper(layers_.size());

        InitData(keeper);

        for (size_t e = 0; e < epochs; ++e) {
            if (shuffle) {
                data_loader.Shuffle();
            }
            optimizer.Optimize(*this, data_loader, loss_function);
            if (debug) {
                std::cerr << "Epoch: " << e << ", Loss: " << Score(data_loader, loss_function) << std::endl;
            }
        }

        SetNullData();
    }
    double Score(const DataLoader& data_loader, const LossFunction& loss_function) const;
    Matrix Predict(Matrix X) const;
    const std::vector<Layer>& Layers() const;

private:
    struct DataKeeper {
        explicit DataKeeper(size_t layers_cnt) : ptr_(std::make_unique<Layer::Data[]>(layers_cnt)) {
        }
        std::unique_ptr<Layer::Data[]> ptr_;
    };

    void InitData(DataKeeper& data_keeper);
    void SetNullData();
    Matrix Apply(Matrix X) const;
    Matrix Propagate(Matrix&& X);
    void BackPropagate(Matrix Y);

    std::vector<Layer> layers_;
};

}  // namespace NNFS
