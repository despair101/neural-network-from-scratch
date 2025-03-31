#include <catch2/catch_test_macros.hpp>

#include "network.h"
#include "utils.h"
#include "adam.h"
#include "sgd.h"

double approx_eq(double x, double y, double eps = 1e-5) {
    return std::abs(x - y) < eps;
}

TEST_CASE("ReadMatrixCSV") {
    using namespace NNFS;
    auto M = ReadMatrixCSV("../data/tests/matrix.csv", false);
    REQUIRE(M == Matrix{{1, 2, 3}, {6, 7, 9}});
    M = ReadMatrixCSV("../data/tests/matrix.csv", true);
    REQUIRE(M == Matrix{{6, 7, 9}});
}

TEST_CASE("ProbsToClass") {
    using namespace NNFS;
    Matrix v{{0.3, 0.3, 0.4}};
    REQUIRE(ProbsToClass(v) == std::vector<Index>{2});
    Matrix u{{0.1, 0, 0.7, 0.2}, {0.29, 0.1, 0.3, 0.31}};
    REQUIRE(ProbsToClass(u) == std::vector<Index>{2, 3});
}

TEST_CASE("Normalize") {
    using namespace NNFS;
    Matrix X{{3, 6}, {4, 1}, {5, 11}};
    X = Normalize(X);
    REQUIRE(approx_eq(X(0, 0), 0));
    REQUIRE(approx_eq(X(1, 0), 0.5));
    REQUIRE(approx_eq(X(2, 0), 1));
    REQUIRE(approx_eq(X(0, 1), 0.5));
    REQUIRE(approx_eq(X(1, 1), 0));
    REQUIRE(approx_eq(X(2, 1), 1));
}

TEST_CASE("Layer") {
    using namespace NNFS;
    Vector x{{3}, {14}};
    Matrix A{{1, 2}, {3, 4}, {5, 6}};
    Vector b{{2}, {6}, {13}};
    Layer layer{A, b, Id()};
    REQUIRE(layer.InputDim() == 2);
    REQUIRE(layer.OutputDim() == 3);
    REQUIRE(layer.Propagate(x) == A * x + b);
}

TEST_CASE("LossFunction") {
    using namespace NNFS;
    Vector a{{0.1}, {0.9}}, b{{0.2}, {0.8}};
    REQUIRE(approx_eq(MSE().Score(a, b), 0.02));
    REQUIRE(approx_eq(CrossEntropy().Score(a, b), 0.180886));
}

TEST_CASE("DataLoader") {
    using namespace NNFS;
    auto X = ReadMatrixCSV("../data/mnist/X_small.csv", false);
    auto Y = ReadMatrixCSV("../data/mnist/Y_small.csv", false);
    for (Index batch_size = 1; batch_size < 20; ++batch_size) {
        DataLoader loader(X, Y, batch_size, 0, true);
        for (auto [x, y] : loader) {
            REQUIRE(x.cols() == batch_size);
            REQUIRE(y.cols() == batch_size);
        }
    }
    DataLoader loader(X, Y, 0);
    size_t cnt = 0;
    for (auto [x, y] : loader) {
        REQUIRE(x.cols() == X.rows());
        REQUIRE(y.cols() == Y.rows());
        ++cnt;
    }
    REQUIRE(cnt == 1);
}

TEST_CASE("SGD") {
    using namespace NNFS;
    auto X = ReadMatrixCSV("../data/mnist/X_small.csv", false);
    auto Y = ReadMatrixCSV("../data/mnist/Y_small.csv", false);
    X = Normalize(X);
    Network network{{784, 128, 10}, {Sigmoid(), SoftMax()}};
    DataLoader loader(X, Y, 16);
    SGD opt(1e-2);
    network.Train(loader, opt, CrossEntropy(), 50);
    double acc = Accuracy(ProbsToClass(Y), ProbsToClass(network.Predict(X)));
    REQUIRE(acc >= 0.8);
}

TEST_CASE("ADAM") {
    using namespace NNFS;
    auto X = ReadMatrixCSV("../data/mnist/X_small.csv", false);
    auto Y = ReadMatrixCSV("../data/mnist/Y_small.csv", false);
    X = Normalize(X);
    Network network{{784, 128, 10}, {Sigmoid(), SoftMax()}};
    DataLoader loader(X, Y, 16);
    ADAM opt(1e-2, 0.9, 0.99, 1e-7);
    network.Train(loader, opt, CrossEntropy(), 10);
    double acc = Accuracy(ProbsToClass(Y), ProbsToClass(network.Predict(X)));
    REQUIRE(acc >= 0.8);
}

TEST_CASE("Simple approximation") {
    using namespace NNFS;
    Matrix X{{{1, 3}, {2, 5}, {3, 7}, {4, 9}}};
    Matrix Y = 2 * X;
    Network network{{2, 2}, {Id()}};
    DataLoader loader(X, Y);
    ADAM optimizer(1e-2, 0.9, 0.99, 1e-8);
    network.Train(loader, optimizer, MSE(), 1000);
    REQUIRE(network.Score(loader, MSE()) < 0.1);
}

TEST_CASE("0.95 accuracy on MNIST (Multilcass classification)") {
    using namespace NNFS;
    auto X = ReadMatrixCSV("../data/mnist/X.csv", false);
    auto Y = ReadMatrixCSV("../data/mnist/Y.csv", false);
    auto [X_train, X_test, Y_train, Y_test] = TrainTestSplit(X, Y, 0.8);
    X_train = Normalize(X_train), X_test = Normalize(X_test);
    ADAM opt(1e-3, 0.9, 0.999, 1e-8);
    DataLoader loader(X_train, Y_train, 16);
    std::vector<Index> dimensions = {784, 128, 10};
    std::vector<std::pair<std::vector<ActivationFunction>, LossFunction>> architectures = {
        {{Sigmoid(), SoftMax()}, CrossEntropy()},
        {{ReLU(), SoftMax()}, CrossEntropy()},
        {{Tanh(), SoftMax()}, CrossEntropy()},
        {{Sigmoid(), Sigmoid()}, MSE()}};
    for (const auto& [layers, loss_function] : architectures) {
        Network network{dimensions, layers};
        network.Train(loader, opt, loss_function, 10);
        double train_acc = Accuracy(ProbsToClass(Y_train), ProbsToClass(network.Predict(X_train)));
        double test_acc = Accuracy(ProbsToClass(Y_test), ProbsToClass(network.Predict(X_test)));
        REQUIRE(train_acc >= 0.97);
        REQUIRE(test_acc >= 0.95);
    }
}

TEST_CASE("0.98 accuracy on RiceTypeClassification (Binary Classification)") {
    using namespace NNFS;
    auto X = ReadMatrixCSV("../data/rice/X.csv", true);
    auto Y = ReadMatrixCSV("../data/rice/Y.csv", true);
    auto [X_train, X_test, Y_train, Y_test] = TrainTestSplit(X, Y, 0.8);
    X_train = Normalize(X_train), X_test = Normalize(X_test);
    ADAM opt(1e-3, 0.9, 0.999, 1e-8);
    DataLoader loader(X_train, Y_train, 32);
    Network network{{10, 64, 2}, {ReLU(), SoftMax()}};
    network.Train(loader, opt, CrossEntropy(), 10);
    double train_acc = Accuracy(ProbsToClass(Y_train), ProbsToClass(network.Predict(X_train)));
    double test_acc = Accuracy(ProbsToClass(Y_test), ProbsToClass(network.Predict(X_test)));
    REQUIRE(train_acc >= 0.98);
    REQUIRE(test_acc >= 0.98);
}

TEST_CASE("Performance test on MNIST") {
    using namespace NNFS;
    auto X = ReadMatrixCSV("../data/mnist/X.csv", false);
    auto Y = ReadMatrixCSV("../data/mnist/Y.csv", false);
    X = Normalize(X);
    {
        Network network{{784, 128, 10}, {ReLU(), SoftMax()}};
        DataLoader loader(X, Y);
        SGD opt(1e-3);
        auto start_time = std::chrono::high_resolution_clock::now();
        network.Train(loader, opt, CrossEntropy(), 20);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cerr << "Average training time for epoch on MNIST dataset (GD on all dataset) is "
                  << std::chrono::duration<double>(end_time - start_time).count() / 20 << std::endl;
    }
    {
        Network network{{784, 128, 10}, {ReLU(), SoftMax()}};
        DataLoader loader(X, Y, 16);
        SGD opt(1e-3);
        auto start_time = std::chrono::high_resolution_clock::now();
        network.Train(loader, opt, CrossEntropy(), 20);
        auto end_time = std::chrono::high_resolution_clock::now();
        std::cerr << "Average training time for epoch on MNIST dataset (SGD with batch_size=16) is "
                  << std::chrono::duration<double>(end_time - start_time).count() / 20 << std::endl;
    }
}
