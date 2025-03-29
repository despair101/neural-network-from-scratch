#pragma once

#include "linalg.h"
#include <vector>

namespace NNFS {

class SGD;
class ADAM;

struct DataSplit {
    Matrix X_train_;
    Matrix X_test_;
    Matrix Y_train_;
    Matrix Y_test_;
};

void ReorderRows(Matrix& A, std::vector<Index> ord);
void ReorderCols(Matrix& A, std::vector<Index> ord);
Matrix ReadMatrixCSV(const std::string& path, bool labels);
DataSplit TrainTestSplit(Matrix X, Matrix Y, double train_frac = 0.66, unsigned seed = 0);
Matrix Normalize(Matrix X);
Index Argmax(RowVector X);
std::vector<Index> ProbsToClass(Matrix Y);

template <typename T>
double Accuracy(const std::vector<T>& y_true, const std::vector<T>& y_pred) {
    assert(y_true.size() == y_pred.size());
    assert(!y_true.empty());
    size_t eq_cnt = 0;
    for (size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) {
            ++eq_cnt;
        }
    }
    return static_cast<double>(eq_cnt) / y_true.size();
}

}  // namespace NNFS
