#include <cassert>

#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/random.h>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

template<typename Value> constexpr auto epsilon() { return epsilon<enoki::scalar_t<Value>>(); }
template<> constexpr auto epsilon<float>() { return 1e-5f; }
template<> constexpr auto epsilon<double>() { return 1e-12; }

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
void cholesky(
    const enoki::Matrix<Value, Size>& in,
    enoki::Matrix<Value, Size>& out,
    enoki::mask_t<Value>& is_psd
) {
    using Matrix = enoki::Matrix<Value, Size>;
    using Mask = enoki::mask_t<Value>;

    out = enoki::zero<Matrix>();
    
    is_psd = Mask(true);
    for(size_t r = 0; r < Size; ++r) {
        Matrix out_t = enoki::transpose(out);
        for(size_t c = 0; c < r; ++c) {
            Value ksum = enoki::zero<Value>();
            for(size_t k = 0; k < c; ++k) {
                ksum += out(r, k) * out(c, k);
            }
            out(r, c) = (in(r, c) - ksum) / out(c, c);
        }
        Value row_sum = enoki::hsum(out_t.col(r) * out_t.col(r));
        Value value = in(r, r) - row_sum;
        is_psd = is_psd && (value > 0);
        out(r, r) = enoki::sqrt(value);
    }
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

void test_cholesky() {
    using Vector3f = Eigen::Matrix<double, 3, 1>;
    using Matrix3f = Eigen::Matrix<double, 3, 3>;
    Matrix3f mat;
    mat <<
        0.0352478, 0.294885, 0.13223, 
        0.294885, 2.46774, 1.10656, 
        0.13223, 1.10656, 0.496206;

    Eigen::LLT<Matrix3f> llt(mat);
    Matrix3f result = llt.matrixL();
    // std::cout << result << std::endl;
}

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
    matrix = enoki::transpose(matrix) * matrix + epsilon<Matrix>(); 

    Matrix chol;
    Mask is_psd;
    cholesky(matrix, chol, is_psd);
    spdlog::info("chol={}", chol);
    test_cholesky();
    return 0;
}
