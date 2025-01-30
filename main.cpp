#include "network.h"
#include <iostream>

int main() {
    Matrix X = Matrix{{1, 2, 3, 4}}.transpose();
    Matrix Y = 2 * X + Vector::Ones(4);
    Network network{{1, 3, 3, 3, 1},
                    {Functions::Sigmoid(), Functions::Sin(), Functions::ReLU(), Functions::Id()},
                    LossFunction()};
    network.Train(X, Y);
    std::cout << "Predicted Y:" << '\n';
    std::cout << network.Predict(X) << '\n';
    std::cout << "True Y:" << '\n';
    std::cout << Y << '\n';
    std::cout << "Loss:" << '\n';
    std::cout << network.Score(Y, network.Predict(X)) << '\n';
}