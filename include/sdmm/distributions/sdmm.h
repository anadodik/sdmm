#pragma once

#include <cassert>

#include <enoki/array.h>
#include <enoki/matrix.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/constants.h"
#include "sdmm/core/utils.h"
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

template<
    typename Vector_,
    typename Matrix_
    // typename Vector=enoki::expr_t<Vector_>,
    // typename Matrix=enoki::expr_t<Matrix_>
>
struct SDMM {
    static_assert(
        std::is_same_v<enoki::scalar_t<Vector_>, enoki::scalar_t<Matrix_>>
    );

    static constexpr size_t MeanSize = enoki::array_size_v<Vector_>;
    static constexpr size_t CovSize = enoki::array_size_v<Matrix_>;

    using Scalar = enoki::value_t<Vector_>;
    using Vector = Vector_;
    using Matrix = Matrix_;
    using Mask = enoki::mask_t<Scalar>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using VectorExpr = enoki::expr_t<Vector>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using VectorS = sdmm::Vector<ScalarS, MeanSize>;
    using MatrixS = sdmm::Matrix<ScalarS, CovSize>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    void prepare();

    // void posterior(const VectorS& point, Value& posterior) const;

    // void pdf(const VectorS& point, Scalar& pdf) const;
    
    void pdf_gaussian(const VectorS& point, Scalar& pdf) const;

    VectorExpr to_standard_normal(const Vector& point) const;

    VectorExpr euclidian_log_map(const VectorS& embedded) const;

    // Make sure to update the ENOKI_STRUCT and ENOKI_STRUCT_SUPPORT
    // declarations when modifying these variables.
    Scalar weight;
    Vector mean;
    Matrix cov;

    Matrix cov_sqrt;
    Scalar inv_cov_sqrt_det;
    Mask cov_is_psd;

    ENOKI_STRUCT(
        SDMM,

        weight,
        mean,
        cov,
        cov_sqrt,
        inv_cov_sqrt_det,
        cov_is_psd
    );
};

template<typename Vector_, typename Matrix_>
auto SDMM<Vector_, Matrix_>::prepare() -> void {
    sdmm::linalg::cholesky(cov, cov_sqrt, cov_is_psd);
    inv_cov_sqrt_det = 1.f / enoki::hprod(enoki::diag(cov_sqrt));
    assert(enoki::all(cov_is_psd));
}

template<typename Vector_, typename Matrix_>
auto SDMM<Vector_, Matrix_>::euclidian_log_map(
    const VectorS& embedded
) const -> VectorExpr {
    return embedded - mean;
}


template<typename Vector_, typename Matrix_>
auto SDMM<Vector_, Matrix_>::to_standard_normal(
    const Vector& point
) const -> VectorExpr {
    VectorExpr standardized;
    return sdmm::linalg::solve(cov_sqrt, point);
}

template<typename Vector_, typename Matrix_>
auto SDMM<Vector_, Matrix_>::pdf_gaussian(
    const VectorS& point, Scalar& pdf
) const -> void {
    VectorExpr tangent = euclidian_log_map(point);
    VectorExpr standardized = to_standard_normal(tangent);
    ScalarExpr squared_norm = enoki::hsum(standardized * standardized);
    pdf =
        inv_cov_sqrt_det *
        gaussian_normalization<ScalarS, CovSize> *
        enoki::exp(ScalarS(-0.5) * squared_norm);
}

// template<typename Value, size_t MeanSize, size_t CovSize>
// void SDMM<Value, MeanSize, CovSize>::pdf(
//     const VectorS& point, Scalar& pdf
// ) const {
//     Value component_pdf;
//     pdf_gaussian(point, pdf);
//     pdf = enoki::hsum(weight * component_pdf);
// }
// 
// template<typename Value, size_t MeanSize, size_t CovSize>
// void SDMM<Value, MeanSize, CovSize>::posterior(
//     const VectorS& point, Value& posterior
// ) const {
//     pdf_gaussian(point, posterior);
//     posterior *= weight;
// }
// 

}

ENOKI_STRUCT_SUPPORT(sdmm::SDMM, weight, mean, cov, cov_sqrt, inv_cov_sqrt_det, cov_is_psd)
