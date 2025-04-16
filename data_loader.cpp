#include "data_loader.h"
#include "utils.h"

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace NNFS {

DataLoader::DataLoader(const Matrix& X, const Matrix& Y, Index batch_size, unsigned seed, bool drop_rest)
    : X_(X.transpose()), Y_(Y.transpose()), gen_(seed), batch_size_(batch_size), drop_rest_(drop_rest) {
    assert(batch_size_ >= 0);
    if (batch_size_ == 0) {
        batch_size_ = X_.cols();
    }
    if (batch_size_ == 0) {
        batches_cnt_ = 0;
    } else if (drop_rest_) {
        batches_cnt_ = X_.cols() / batch_size_;
    } else {
        batches_cnt_ = (X_.cols() + batch_size_ - 1) / batch_size_;
    }
}

DataLoader::BatchIterator DataLoader::begin() const {
    return {0, batch_size_ * batches_cnt_, batch_size_, X_, Y_};
}

DataLoader::BatchIterator DataLoader::end() const {
    return {batch_size_ * batches_cnt_, batch_size_ * batches_cnt_, batch_size_, X_, Y_};
}

void DataLoader::Shuffle() {
    std::vector<Index> ord(X_.cols());
    std::iota(ord.begin(), ord.end(), 0u);
    std::shuffle(ord.begin(), ord.end(), gen_);
    ReorderCols(X_, ord);
    ReorderCols(Y_, ord);
}

DataLoader::BatchIterator::BatchIterator(Index ind, Index size, Index batch_size, const Matrix& X, const Matrix& Y)
    : ind_(ind), size_(size), batch_size_(batch_size), X_(&X), Y_(&Y) {
}

DataLoader::Batch DataLoader::BatchIterator::operator*() const {
    assert(ind_ < size_);
    Index true_batch_size = std::min(batch_size_, size_ - ind_);
    return {
        X_->block(0, ind_, X_->rows(), true_batch_size),
        Y_->block(0, ind_, Y_->rows(), true_batch_size),
    };
}

DataLoader::BatchIterator& DataLoader::BatchIterator::operator++() {
    assert(ind_ < size_);
    Index true_batch_size = std::min(batch_size_, size_ - ind_);
    ind_ += true_batch_size;
    return *this;
}

bool DataLoader::BatchIterator::operator!=(const BatchIterator& other) const {
    return ind_ != other.ind_;
}

bool DataLoader::BatchIterator::operator==(const BatchIterator& other) const {
    return ind_ == other.ind_;
}

}  // namespace NNFS
