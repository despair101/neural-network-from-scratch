#include "utils.h"

#include <numeric>
#include <random>
#include <vector>
#include <fstream>

namespace NNFS {

void ReorderRows(Matrix& A, std::vector<Index> ord) {
    assert(A.rows() == ord.size());
    for (Index i = 0; i < ord.size(); ++i) {
        while (ord[i] != i) {
            A.row(i).swap(A.row(ord[i]));
            std::swap(ord[i], ord[ord[i]]);
        }
    }
}

void ReorderCols(Matrix& A, std::vector<Index> ord) {
    assert(A.cols() == ord.size());
    for (Index i = 0; i < ord.size(); ++i) {
        while (ord[i] != i) {
            A.col(i).swap(A.col(ord[i]));
            std::swap(ord[i], ord[ord[i]]);
        }
    }
}

DataSplit TrainTestSplit(Matrix X, Matrix Y, double train_frac, unsigned seed) {
    assert(X.rows() == Y.rows());
    assert(train_frac >= 0 && train_frac <= 1);
    Index rows = X.rows();
    std::vector<Index> ord(rows);
    std::mt19937 gen(seed);
    std::iota(ord.begin(), ord.end(), 0u);
    std::shuffle(ord.begin(), ord.end(), gen);
    ReorderRows(X, ord);
    ReorderRows(Y, ord);

    Index train_rows = std::min(static_cast<Index>(rows * train_frac), rows);
    Index test_rows = rows - train_rows;
    return {.X_train_ = X.block(0, 0, train_rows, X.cols()),
            .X_test_ = X.block(train_rows, 0, test_rows, X.cols()),
            .Y_train_ = Y.block(0, 0, train_rows, Y.cols()),
            .Y_test_ = Y.block(train_rows, 0, test_rows, Y.cols())};
}

Matrix ReadMatrixCSV(const std::string& path, bool labels) {
    std::ifstream in(path);
    std::vector<std::vector<double>> data;
    std::string line;
    if (labels) {
        std::getline(in, line);
    }
    while (std::getline(in, line)) {
        data.push_back({});
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ',')) {
            data.back().push_back(std::stod(item));
        }
    }
    assert(!data.empty());
    size_t rows = data.size(), cols = data.front().size();
    assert(rows < LONG_MAX && cols < LONG_MAX);
    for (Index i = 0; i < rows; ++i) {
        assert(data[i].size() == cols);
    }
    Matrix M(rows, cols);
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) {
            M(i, j) = data[i][j];
        }
    }
    return M;
}

Matrix Normalize(Matrix X) {
    for (Index i = 0; i < X.cols(); ++i) {
        double min = X.col(i).minCoeff(), max = X.col(i).maxCoeff();
        if (min == max) {
            X.col(i).setZero();
        } else {
            X.col(i) = (X.col(i).array() - min) / (max - min);
        }
    }
    return X;
}

Index Argmax(RowVector x) {
    Index ind_max = 0;
    for (Index i = 0; i < x.cols(); ++i) {
        if (x(i) > x(ind_max)) {
            ind_max = i;
        }
    }
    return ind_max;
}

std::vector<Index> ProbsToClass(Matrix Y) {
    std::vector<Index> res(Y.rows());
    for (Index i = 0; i < Y.rows(); ++i) {
        res[i] = Argmax(Y.row(i));
    }
    return res;
}

}  // namespace NNFS
