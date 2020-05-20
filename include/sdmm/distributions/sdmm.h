#pragma once

#include <cassert>

#include <enoki/array.h>
#include <enoki/matrix.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/constants.h"
#include "sdmm/linalg/cholesky.h"

namespace sdmm {

template<typename T>
using vector_t = typename T::Vector;

template<typename T>
using matrix_t = typename T::Matrix;

template<typename T>
using vector_s_t = typename T::VectorS;

template<typename T>
using matrix_s_t = typename T::MatrixS;

template<typename Value, size_t MeanSize, size_t CovSize>
struct SDMM {
    using VectorS = enoki::Array<enoki::scalar_t<Value>, MeanSize>;
    using MatrixS = enoki::Matrix<enoki::scalar_t<Value>, CovSize>;

    using Vector = enoki::Array<Value, MeanSize>;
    using Matrix = enoki::Matrix<Value, CovSize>;
    using Mask = enoki::mask_t<Value>;

    SDMM() = default;
    SDMM(SDMM&& other) = default;
    SDMM(const SDMM& other) = default;
    SDMM& operator=(SDMM&& other) = default;
    SDMM& operator=(const SDMM& other) = default;

    void zero() {
        weight = enoki::zero<Value>();
        mean = enoki::zero<Vector>();
        cov = enoki::zero<Matrix>();
    }

    void prepare() {
        sdmm::linalg::cholesky(cov, cov_sqrt, cov_is_psd);
        inv_cov_sqrt_det = Value(1) / enoki::hprod(enoki::diag(cov_sqrt));
        assert(enoki::all(cov_is_psd));
    }
    
    void pdf_gaussian(const VectorS& point, Value& pdf) const;

    void to_standard_normal(const Vector& point, Vector& standardized) const {
        sdmm::linalg::solve(cov_sqrt, point, standardized);
    }

    Vector euclidian_log_map(const Vector& embedded) const {
        return embedded - mean;
    }

    Value weight;
    Vector mean;
    Matrix cov;

    Matrix cov_sqrt;
    Value inv_cov_sqrt_det;
    Mask cov_is_psd;
};

template<typename Value, size_t MeanSize, size_t CovSize>
void SDMM<Value, MeanSize, CovSize>::pdf_gaussian(
    const SDMM<Value, MeanSize, CovSize>::VectorS& point, Value& pdf
) const {
    Vector standardized;
    to_standard_normal(euclidian_log_map(point), standardized);
    Value squared_norm = enoki::hsum(
        standardized * standardized
    );

    pdf =
        inv_cov_sqrt_det *
        gaussian_normalization<enoki::scalar_t<Value>, CovSize> *
        enoki::exp(Value(-0.5) * squared_norm);
}

}
