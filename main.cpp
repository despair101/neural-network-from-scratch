#include "network.h"

#include <iostream>

int main() {
    using namespace NeuralNetworkFromScratch;
    Matrix X = Matrix{{1, 2, 3, 4}};
    Matrix Y = 2 * X + RowVector::Ones(4);
    Network network{{1, 8, 8, 8, 1}, {Sin(), Sigmoid(), ReLU(), Id()}};
    network.Train(X, Y, 1000, 0.01, LossFunction());
    std::cout << "Predicted Y:" << '\n';
    std::cout << network.Predict(X) << '\n';
    std::cout << "True Y:" << '\n';
    std::cout << Y << '\n';
    std::cout << "Loss:" << '\n';
    std::cout << LossFunction().Score(Y, network.Predict(X)) << '\n';
}
