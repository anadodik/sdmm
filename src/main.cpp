#include <cassert>

#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/random.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/linalg/epsilon.h"
#include "sdmm/linalg/cholesky.h"

template<typename Value, size_t Size>
enoki::Array<Value, Size> row(const enoki::Matrix<Value, Size>& matrix, int row) {
    using Array = enoki::Array<Value, Size>;
    using Index = enoki::Array<uint32_t, Size>;

    Index index = enoki::arange<Index>() * Size;
    void* mem = (void*) (matrix.coeff(0).data() + row);
    Array result;
    for (size_t i = 0; i < Array::Size; ++i)
        result[i] = ((Value *) mem)[index[i]];
    return result;
}

template<typename Value, size_t Size>
class MatrixL : enoki::Matrix<Value, Size> {
};

template<typename Value, size_t Size>
class AdjointMatrix : enoki::Matrix<Value, Size> {
};

template<typename ScalarT, int MeanSize, int CovSize>
struct SDMMParams {
    using Scalar = enoki::Array<ScalarT, 64>;
    using Vector = enoki::Array<Scalar, MeanSize>;
    using Matrix = enoki::Matrix<Scalar, CovSize>;

    SDMMParams(SDMMParams&& other) = default;
    SDMMParams(const SDMMParams& other) = default;
    SDMMParams& operator=(SDMMParams&& other) = default;
    SDMMParams& operator=(const SDMMParams& other) = default;

    void zero() {
        weights = enoki::zero<Scalar>();
        means = enoki::zero<Vector>();
        covs = enoki::zero<Matrix>();
    }

    Vector toStandardNormal(const Vector& sample) const {
        return chol_inv * sample;
    }


    Scalar weights;
    Vector means;
    Matrix covs;
    Matrix chol;
    Matrix chol_inv;
};

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    // enoki::set_flush_denormals(true);
    spdlog::info("Hello SDMM! Using max packet size: {}", enoki::max_packet_size);
    using Array = enoki::Array<float, 2>;
    using Matrix = enoki::Matrix<Array, 3>;
    using Mask = enoki::mask_t<Array>;
    using RNG = enoki::PCG32<Array>;
    RNG rng;
    Matrix matrix = enoki::zero<Matrix>();
    for(int c = 0; c < 3; ++c) {
        matrix.col(c) = rng.next_float32();
    }
    matrix = enoki::transpose(matrix) * matrix + sdmm::linalg::epsilon<Matrix>(); 

    Matrix chol;
    Mask is_psd;
    sdmm::linalg::cholesky(matrix, chol, is_psd);
    // spdlog::info("matrix={}", matrix);
    auto chol_diff = enoki::abs(chol * enoki::transpose(chol) - matrix) < sdmm::linalg::epsilon<float>();
    spdlog::info("chol_diff={}", chol_diff);
    return 0;
}
