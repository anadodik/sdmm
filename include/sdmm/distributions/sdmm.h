#pragma once

#include <cassert>

#include <enoki/array.h>
#include <enoki/dynamic.h>
#include <enoki/matrix.h>
#include <enoki/random.h>

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
using matrix_t = typename T::Matrix;

template<typename T>
using matrix_expr_t = typename T::MatrixExpr;

template<typename T>
using matrix_s_t = typename T::MatrixS;

template<typename T>
using tangent_t = typename T::Tangent;

template<typename T>
using tangent_s_t = typename T::TangentS;

template<typename T>
using tangent_expr_t = typename T::TangentExpr;

template<typename T>
using embedded_t = typename T::Embedded;

template<typename T>
using embedded_s_t = typename T::EmbeddedS;

template<typename T>
using embedded_expr_t = typename T::EmbeddedExpr;

template<typename SDMM, typename T>
using replace_embedded_t = sdmm::Vector<T, SDMM::MeanSize>;

template<typename SDMM, typename T>
using replace_tangent_t = sdmm::Vector<T, SDMM::CovSize>;

template<typename Matrix_, typename TangentSpace_>
struct SDMM {
    static_assert(
        std::is_same_v<
            enoki::scalar_t<tangent_t<TangentSpace_>>, enoki::scalar_t<Matrix_>
        >
    );
    static_assert(tangent_t<TangentSpace_>::Size == Matrix_::Size);

    using TangentSpace = TangentSpace_;
    using Tangent = tangent_t<TangentSpace>;
    using Embedded = embedded_t<TangentSpace>;
    using Scalar = enoki::value_t<Tangent>;
    using Mask = enoki::mask_t<Scalar>;
    using Matrix = Matrix_;

    static constexpr size_t MeanSize = enoki::array_size_v<Embedded>;
    static constexpr size_t CovSize = enoki::array_size_v<Tangent>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    using TangentExpr = tangent_expr_t<TangentSpace>;
    using EmbeddedExpr = embedded_expr_t<TangentSpace>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using TangentS = tangent_s_t<TangentSpace>;
    using EmbeddedS = embedded_s_t<TangentSpace>;
    using MatrixS = sdmm::Matrix<ScalarS, CovSize>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    auto prepare_cov() -> void;

    template<typename RNG, typename EmbeddedIn, typename TangentIn>
    auto sample(RNG& rng, EmbeddedIn& sample, Scalar& inv_jacobian, TangentIn& tangent) const -> void;

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
template<typename Matrix_, typename TangentSpace_>
auto SDMM<Matrix_, TangentSpace_>::prepare_cov() -> void {
    sdmm::linalg::cholesky(cov, cov_sqrt, cov_is_psd);
    inv_cov_sqrt_det = 1.f / enoki::hprod(enoki::diag(cov_sqrt));
    assert(enoki::all(cov_is_psd));
}

template<typename Value>
auto box_mueller_transform(const Value& u1, const Value& u2)
        -> std::pair<enoki::expr_t<Value>, enoki::expr_t<Value>> {
    using ValueExpr = enoki::expr_t<Value>;
    ValueExpr radius = enoki::sqrt(-2 * enoki::log(1 - u1));
    ValueExpr theta = 2 * M_PI * u2;
    auto [sin, cos] = enoki::sincos(theta);
    return {sin * radius, cos * radius};
}

template<typename Matrix_, typename TangentSpace_>
template<typename RNG, typename EmbeddedIn, typename TangentIn>
auto SDMM<Matrix_, TangentSpace_>::sample(
    RNG& rng, EmbeddedIn& sample, Scalar& inv_jacobian, TangentIn& tangent
) const -> void {
    auto weight_inv_sample = rng.next_float32();
    using Float32 = typename RNG::Float32;
    using UInt32 = typename RNG::UInt32;
    UInt32 gaussian_idx = enoki::binary_search(
        0,
        enoki::slices(weight.cdf),
        [&](UInt32 index) {
            return enoki::gather<Float32>(weight.cdf, index) < weight_inv_sample;
        }
    );

    if(enoki::slices(tangent) != enoki::slices(gaussian_idx)) {
        enoki::set_slices(tangent, enoki::slices(gaussian_idx)); 
    }
    for(size_t dim_i = 0; dim_i < CovSize; dim_i += 2) {
        auto [u1, u2] = box_mueller_transform(
            rng.next_float32(), rng.next_float32()
        );
        tangent.coeff(dim_i) = u1;
        if(dim_i + 1 < CovSize) {
            tangent.coeff(dim_i + 1) = u2;
        }
    }

    Matrix sampled_cov_sqrt;
    enoki::set_slices(sampled_cov_sqrt, enoki::slices(gaussian_idx));
    for(size_t mat_i = 0; mat_i < enoki::slices(gaussian_idx); ++mat_i) {
        uint32_t index = gaussian_idx.coeff(mat_i);
        enoki::slice(sampled_cov_sqrt, mat_i) = enoki::slice(cov_sqrt, index);
    }
    // TODO: ^gather
    // auto covs = enoki::gather<Matrix, sizeof(MatrixS)>(cov_sqrt.data(), weight_indices);
    tangent = sampled_cov_sqrt * tangent;
    if(enoki::slices(sample) != enoki::slices(gaussian_idx)) {
        enoki::set_slices(sample, enoki::slices(gaussian_idx)); 
    }
    if(enoki::slices(inv_jacobian) != enoki::slices(gaussian_idx)) {
        enoki::set_slices(inv_jacobian, enoki::slices(gaussian_idx)); 
    }
    for(size_t ts_i = 0; ts_i < enoki::slices(gaussian_idx); ++ts_i) {
        uint32_t index = gaussian_idx.coeff(ts_i);
        enoki::slice(sample, ts_i) =
            enoki::slice(tangent_space, index).from(
                enoki::slice(tangent, ts_i), enoki::slice(inv_jacobian, ts_i)
            );
    }
}

template<typename Matrix_, typename TangentSpace_>
auto SDMM<Matrix_, TangentSpace_>::to_standard_normal(
    const Tangent& point
) const -> TangentExpr {
    TangentExpr standardized;
    return sdmm::linalg::solve(cov_sqrt, point);
}

template<typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn, typename TangentIn>
auto SDMM<Matrix_, TangentSpace_>::pdf_gaussian(
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

template<typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn>
auto SDMM<Matrix_, TangentSpace_>::pdf_gaussian(
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

template<typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn, typename TangentIn>
auto SDMM<Matrix_, TangentSpace_>::posterior(
    const EmbeddedIn& point, Scalar& posterior, TangentIn& tangent
) const -> void {
    pdf_gaussian(point, posterior, tangent);
    posterior *= weight.pmf;
}

template<typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn>
auto SDMM<Matrix_, TangentSpace_>::posterior(
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
