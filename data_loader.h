#pragma once

#include "linalg.h"

#include <random>

namespace NNFS {

class DataLoader {
public:
    struct Batch {
        Matrix X_;
        Matrix Y_;
    };

    class BatchIterator {
    public:
        BatchIterator(Index ind, Index size, Index batch_size, const Matrix& X, const Matrix& Y);
        Batch operator*() const;
        BatchIterator& operator++();
        bool operator!=(const BatchIterator& other) const;

    private:
        Index ind_;
        Index size_;
        Index batch_size_;
        const Matrix& X_;
        const Matrix& Y_;
    };

    DataLoader(const Matrix& X, const Matrix& Y, Index batch_size = 0, unsigned seed = 0, bool drop_rest = true);
    BatchIterator begin() const;
    BatchIterator end() const;
    void Shuffle();

private:
    Matrix X_;
    Matrix Y_;
    std::mt19937_64 gen_;
    Index batch_size_;
    Index batches_cnt_;
    bool drop_rest_;
};

}  // namespace NNFS
