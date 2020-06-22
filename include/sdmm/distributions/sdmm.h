#pragma once

#include <cassert>

#include <enoki/array.h>
#include <enoki/dynamic.h>
#include <enoki/matrix.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/constants.h"
#include "sdmm/core/utils.h"
#include "sdmm/distributions/categorical.h"
#include "sdmm/linalg/cholesky.h"
#include "sdmm/spaces/euclidian.h"
#include "sdmm/spaces/directional.h"
#include "sdmm/spaces/spatio_directional.h"

namespace sdmm {

template<typename T>
using tangent_space_t = typename T::TangentSpace;

template<typename T>
using vector_t = typename T::Vector;

template<typename T>
using vector_expr_t = typename T::VectorExpr;

template<typename T>
using vector_s_t = typename T::VectorS;

template<typename T>
using matrix_t = typename T::Matrix;

template<typename T>
using matrix_expr_t = typename T::MatrixExpr;

template<typename T>
using matrix_s_t = typename T::MatrixS;

template<typename T>
using tangent_t = typename T::Tangent;

template<typename T>
using tangent_expr_t = typename T::TangentExpr;

template<typename T>
using embedded_t = typename T::Embedded;

template<typename T>
using embedded_expr_t = typename T::EmbeddedExpr;


// TODO: Vector_ not necessary.
template<typename Vector_, typename Matrix_, typename TangentSpace_>
struct SDMM {
    static_assert(
        std::is_same_v<enoki::scalar_t<Vector_>, enoki::scalar_t<Matrix_>>
    );

    static constexpr size_t MeanSize = enoki::array_size_v<Vector_>;
    static constexpr size_t CovSize = enoki::array_size_v<Matrix_>;

    using TangentSpace = TangentSpace_;
    using Vector = Vector_;
    using Matrix = Matrix_;
    using Tangent = tangent_t<TangentSpace>;
    using Embedded = embedded_t<TangentSpace>;
    using Scalar = enoki::value_t<Vector_>;
    using Mask = enoki::mask_t<Scalar>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using VectorExpr = enoki::expr_t<Vector>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    using TangentExpr = tangent_expr_t<TangentSpace>;
    using EmbeddedExpr = embedded_expr_t<TangentSpace>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using VectorS = sdmm::Vector<ScalarS, MeanSize>;
    using MatrixS = sdmm::Matrix<ScalarS, CovSize>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    auto prepare_cov() -> void;

    template<typename EmbeddedIn, typename TangentIn>
    auto pdf_gaussian(const EmbeddedIn& point, Scalar& pdf, TangentIn& tangent) const -> void;

    template<typename EmbeddedIn, typename TangentIn>
    auto posterior(const EmbeddedIn& point, Scalar& posterior, TangentIn& tangent) const -> void;

    template<typename EmbeddedIn>
    auto pdf_gaussian(const EmbeddedIn& point, Scalar& pdf) const -> void;

    template<typename EmbeddedIn>
    auto posterior(const EmbeddedIn& point, Scalar& posterior) const -> void;

    auto to_standard_normal(const Tangent& point) const -> TangentExpr;

    // Make sure to update the ENOKI_STRUCT and ENOKI_STRUCT_SUPPORT
    // declarations when modifying these variables.
    Categorical<Scalar> weight;
    TangentSpace tangent_space;
    Matrix cov;

    // TODO: make struct Cholesky {};
    Matrix cov_sqrt;
    Scalar inv_cov_sqrt_det;
    Mask cov_is_psd;

    ENOKI_STRUCT(
        SDMM,

        weight,
        tangent_space,
        cov,

        cov_sqrt,
        inv_cov_sqrt_det,
        cov_is_psd
    );
};

template<typename SDMM>
[[nodiscard]] inline auto prepare(SDMM& distribution) {
    if constexpr(enoki::is_dynamic_v<SDMM>) {
        enoki::vectorize(
            VECTORIZE_WRAP_MEMBER(prepare_cov),
            distribution
        );
    } else {
        distribution.prepare_cov();
    }
    return distribution.weight.prepare();
}

// TODO: make [[nodiscard]] and check cov_is_psd
template<typename Vector_, typename Matrix_, typename TangentSpace_>
auto SDMM<Vector_, Matrix_, TangentSpace_>::prepare_cov() -> void {
    sdmm::linalg::cholesky(cov, cov_sqrt, cov_is_psd);
    inv_cov_sqrt_det = 1.f / enoki::hprod(enoki::diag(cov_sqrt));
    assert(enoki::all(cov_is_psd));
}

template<typename Vector_, typename Matrix_, typename TangentSpace_>
auto SDMM<Vector_, Matrix_, TangentSpace_>::to_standard_normal(
    const Tangent& point
) const -> TangentExpr {
    TangentExpr standardized;
    return sdmm::linalg::solve(cov_sqrt, point);
}

template<typename Vector_, typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn, typename TangentIn>
auto SDMM<Vector_, Matrix_, TangentSpace_>::pdf_gaussian(
    const EmbeddedIn& point, Scalar& pdf, TangentIn& tangent
) const -> void {
    ScalarExpr inv_jacobian;
    tangent = tangent_space.to(point, inv_jacobian);
    TangentExpr standardized = to_standard_normal(tangent);
    ScalarExpr squared_norm = enoki::squared_norm(standardized);
    pdf =
        inv_cov_sqrt_det *
        inv_jacobian *
        gaussian_normalization<ScalarS, CovSize> *
        enoki::exp(ScalarS(-0.5) * squared_norm);
}

template<typename Vector_, typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn>
auto SDMM<Vector_, Matrix_, TangentSpace_>::pdf_gaussian(
    const EmbeddedIn& point, Scalar& pdf
) const -> void {
    // #ifdef NDEBUG
    // spdlog::warn(
    //     "Using allocating call to pdf_gaussian. "
    //     "Consider pre-allocating tangent_vectors."
    // );
    // #endif // NDEBUG
    TangentExpr tangent;
    pdf_gaussian(point, pdf, tangent);
}

template<typename Vector_, typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn, typename TangentIn>
auto SDMM<Vector_, Matrix_, TangentSpace_>::posterior(
    const EmbeddedIn& point, Scalar& posterior, TangentIn& tangent
) const -> void {
    pdf_gaussian(point, posterior, tangent);
    posterior *= weight.pmf;
}

template<typename Vector_, typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn>
auto SDMM<Vector_, Matrix_, TangentSpace_>::posterior(
    const EmbeddedIn& point, Scalar& pdf
) const -> void {
    // #ifdef NDEBUG
    // spdlog::warn(
    //     "Using allocating call to posterior. "
    //     "Consider pre-allocating tangent_vectors."
    // );
    // #endif // NDEBUG
    TangentExpr tangent;
    posterior(point, pdf, tangent);
}


// template<typename Value, size_t MeanSize, size_t CovSize>
// void SDMM<Value, MeanSize, CovSize>::pdf(
//     const VectorS& point, Scalar& pdf
// ) const {
//     Value component_pdf;
//     pdf_gaussian(point, pdf);
//     pdf = enoki::hsum(weight * component_pdf);
// }

}

ENOKI_STRUCT_SUPPORT(
    sdmm::SDMM,

    weight,
    tangent_space,
    cov,
    
    cov_sqrt,
    inv_cov_sqrt_det,
    cov_is_psd
);
