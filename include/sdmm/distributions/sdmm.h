#pragma once

#include <cassert>

#include <enoki/array.h>
#include <enoki/dynamic.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/constants.h"
#include "sdmm/core/utils.h"
#include "sdmm/distributions/categorical.h"
#include "sdmm/linalg/cholesky.h"

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

    auto prepare_cov() -> MaskExpr;
    auto prepare() -> MaskExpr;

    template<typename RNG, typename EmbeddedIn, typename ScalarIn, typename TangentIn>
    auto sample(RNG& rng, EmbeddedIn& sample, ScalarIn& inv_jacobian, TangentIn& tangent) const -> void;

    template<typename EmbeddedIn, typename TangentIn>
    auto pdf_gaussian(const EmbeddedIn& point, Scalar& pdf, TangentIn& tangent) const -> void;

    template<typename EmbeddedIn, typename TangentIn>
    auto posterior(const EmbeddedIn& point, Scalar& posterior, TangentIn& tangent) const -> void;

    template<typename EmbeddedIn>
    auto pdf_gaussian(const EmbeddedIn& point, Scalar& pdf) const -> void;

    template<typename EmbeddedIn>
    auto posterior(const EmbeddedIn& point, Scalar& posterior) const -> void;

    auto to_standard_normal(const Tangent& point) const -> TangentExpr;

    Categorical<Scalar> weight;
    TangentSpace tangent_space;
    Matrix cov;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(SDMM, weight.pmf, tangent_space.mean, cov)

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
[[nodiscard]] inline auto prepare_vectorized(SDMM& distribution) {
    typename SDMM::Mask cov_success;
    if constexpr(enoki::is_dynamic_v<SDMM>) {
        cov_success = enoki::vectorize(
            VECTORIZE_WRAP_MEMBER(prepare_cov),
            distribution
        );
    } else {
        distribution.prepare_cov();
    }
    return distribution.weight.prepare() && enoki::all(cov_success);
}

template<typename Matrix_, typename TangentSpace_>
auto SDMM<Matrix_, TangentSpace_>::prepare() -> MaskExpr {
    MaskExpr is_psd = prepare_cov();
    weight.pmf &= is_psd;
	if(!weight.prepare()) {
        spdlog::info("all weights 0");
        return is_psd; // MaskExpr(false);
    }
	return is_psd;
}

template<typename Matrix_, typename TangentSpace_>
auto SDMM<Matrix_, TangentSpace_>::prepare_cov() -> MaskExpr {
    sdmm::linalg::cholesky(cov, cov_sqrt, cov_is_psd);
    inv_cov_sqrt_det = 1.f / enoki::hprod(enoki::diag(cov_sqrt));
    bool all_psd = enoki::all(cov_is_psd);
    // if(!all_psd) {
    //     std::cerr << fmt::format("cov={}\n", cov);
    //     std::cerr << fmt::format("all_psd={}\n", cov_is_psd);
    //     assert(all_psd);
    // }
    return cov_is_psd;
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
template<typename RNG, typename EmbeddedIn, typename ScalarIn, typename TangentIn>
auto SDMM<Matrix_, TangentSpace_>::sample(
    RNG& rng, EmbeddedIn& sample, ScalarIn& inv_jacobian, TangentIn& tangent
) const -> void {
    auto weight_inv_sample = rng.next_float32();
    using Float32 = typename RNG::Float32;
    using UInt32 = typename RNG::UInt32;
    UInt32 gaussian_idx = enoki::binary_search(
        0,
        enoki::slices(weight.cdf) - 1,
        // [&](UInt32 index) {
        //     return enoki::gather<Float32>(weight.cdf, index) < weight_inv_sample;
        // }
        [&](UInt32 index) {
            return weight.cdf[index] < weight_inv_sample;
        }
    );
    while(gaussian_idx > 0 && weight.pmf[gaussian_idx] == 0) {
        --gaussian_idx;
    }

    // if(enoki::slices(tangent) != enoki::slices(gaussian_idx)) {
    //     enoki::set_slices(tangent, enoki::slices(gaussian_idx)); 
    // }
    for(size_t dim_i = 0; dim_i < CovSize; dim_i += 2) {
        auto [u1, u2] = box_mueller_transform(
            rng.next_float32(), rng.next_float32()
        );
        tangent.coeff(dim_i) = u1;
        if(dim_i + 1 < CovSize) {
            tangent.coeff(dim_i + 1) = u2;
        }
    }

    auto sampled_cov_sqrt = enoki::slice(cov_sqrt, gaussian_idx);
    // enoki::set_slices(sampled_cov_sqrt, enoki::slices(gaussian_idx));
    // for(size_t mat_i = 0; mat_i < enoki::slices(gaussian_idx); ++mat_i) {
    //     uint32_t index = gaussian_idx.coeff(mat_i);
    //     enoki::slice(sampled_cov_sqrt, mat_i) = enoki::slice(cov_sqrt, index);
    // }
    // TODO: ^gather
    // auto covs = enoki::gather<Matrix, sizeof(MatrixS)>(cov_sqrt.data(), weight_indices);
    tangent = TangentIn(sampled_cov_sqrt * tangent);
    // if(enoki::slices(sample) != enoki::slices(gaussian_idx)) {
    //     enoki::set_slices(sample, enoki::slices(gaussian_idx)); 
    // }
    // if(enoki::slices(inv_jacobian) != enoki::slices(gaussian_idx)) {
    //     enoki::set_slices(inv_jacobian, enoki::slices(gaussian_idx)); 
    // }
    sample = enoki::slice(tangent_space, gaussian_idx).from(tangent, inv_jacobian);
    // for(size_t ts_i = 0; ts_i < enoki::slices(gaussian_idx); ++ts_i) {
    //     uint32_t index = gaussian_idx.coeff(ts_i);
    //     enoki::slice(sample, ts_i) =
    //         enoki::slice(tangent_space, index).from(
    //             enoki::slice(tangent, ts_i), enoki::slice(inv_jacobian, ts_i)
    //         );
    // }
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
    // if(CovSize == 2 && !enoki::all(enoki::isfinite(pdf))) {
    //     std::cerr << fmt::format(
    //         "pdf={}\n"
    //         "point={}\n"
    //         "tangent={}\n"
    //         "standardized={}\n"
    //         "cov_sqrt={}\n"
    //         "inv_cov_sqrt_det={}\n"
    //         "inv_jacobian={}\n",
    //         pdf,
    //         point,
    //         tangent,
    //         standardized,
    //         cov_sqrt,
    //         inv_cov_sqrt_det,
    //         inv_jacobian
    //     );
    // }
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

// TODO: move normalization to this function.
template<typename Matrix_, typename TangentSpace_>
template<typename EmbeddedIn, typename TangentIn>
auto SDMM<Matrix_, TangentSpace_>::posterior(
    const EmbeddedIn& point, Scalar& posterior, TangentIn& tangent
) const -> void {
    pdf_gaussian(point, posterior, tangent);
    posterior *= weight.pmf;
}

// TODO: move normalization to this function.
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

template<
    typename SDMM,
    std::enable_if_t<
        sdmm::embedded_t<SDMM>::Size != sdmm::tangent_t<SDMM>::Size &&
        sdmm::embedded_t<SDMM>::Size == 3,
        int
    > = 0
>
auto product(SDMM& first, SDMM& second, SDMM& result) {
    // TODO: implement single lobe selection from learned BSDF
    using ScalarS = typename SDMM::ScalarS;
    using EmbeddedS = sdmm::embedded_s_t<SDMM>;
    using TangentS = sdmm::tangent_s_t<SDMM>;
    using MatrixS = sdmm::matrix_s_t<SDMM>;

    using ScalarExpr = enoki::expr_t<decltype(enoki::packet(first.weight.pmf, 0))>;
    using MatrixExpr = enoki::expr_t<decltype(enoki::packet(first.cov, 0))>;

    size_t first_size = enoki::slices(first);
    size_t second_size = enoki::slices(second);
    size_t product_size = first_size * second_size;

    if(enoki::slices(result) != product_size) {
        enoki::set_slices(result, product_size);
    }

    size_t first_packets = enoki::packets(first);
    size_t product_packets = enoki::packets(result);

    bool use_jacobian_correction = false;

    for(size_t first_i = 0; first_i < first_packets; ++first_i) {
        for(size_t second_i = 0; second_i < second_size; ++second_i) {
            size_t product_i = first_i * second_size + second_i;

            MatrixExpr first_cov = enoki::packet(first.cov, first_i);
            MatrixS second_cov_original = enoki::slice(second.cov, second_i);

            auto&& first_ts = enoki::packet(first.tangent_space, first_i);
            auto&& second_ts = enoki::slice(second.tangent_space, second_i);

            TangentS first_mean = enoki::zero<TangentS>();
            EmbeddedS second_mean_embedded = second_ts.mean;
            auto from_center_jacobian = second_ts.from_center_jacobian();
            ScalarExpr inv_jacobian;
            auto [second_mean, to_jacobian] = first_ts.to_jacobian(second_mean_embedded, inv_jacobian);
            MatrixExpr J = to_jacobian * from_center_jacobian;

            // spdlog::info("J_to={}", J);
            MatrixExpr second_cov = J * second_cov_original * linalg::transpose(J);
            // spdlog::info("fc={}, first_cov={} vs fc0={}", first.cov, first_cov, enoki::slice(first.cov, 0));
            // spdlog::info("second_cov={}", second_cov);
            MatrixExpr cov_sum = first_cov + second_cov;
            MatrixExpr cov_sum_sqrt;
            enoki::mask_t<ScalarExpr> is_psd;
            linalg::cholesky(cov_sum, cov_sum_sqrt, is_psd);
            if(!enoki::all(is_psd)) {
                // spdlog::info("second_cov_original={}", second_cov_original);
                auto embedded_local = first_ts.coordinate_system.to * second_mean_embedded;
                std::cerr << fmt::format(
                    "({}, {}) = "
                    "is_psd={}, "
                    "second_mean_embedded={}, "
                    "embedded_local={}, "
                    "J_to={}\n",
                    first_i,
                    second_i,
                    is_psd,
                    second_mean_embedded,
                    embedded_local,
                    J
                );

                std::cerr << fmt::format("second_cov={}\n", second_cov);
                // spdlog::info("cov_sum={}", cov_sum);
            }
            MatrixExpr inv_cov_sum_sqrt = linalg::inverse_lower_tri(cov_sum_sqrt);
            // spdlog::info("cov_sum_sqrt={}", cov_sum_sqrt);
            // spdlog::info("inv_cov_sum_sqrt={}", inv_cov_sum_sqrt);

            MatrixExpr first_cov_premult = inv_cov_sum_sqrt * first_cov;
            MatrixExpr second_cov_premult = inv_cov_sum_sqrt * second_cov;
            // spdlog::info("first_cov_premult={}", first_cov_premult);
            // spdlog::info("second_cov_premult={}", second_cov_premult);

            auto first_mean_premult = inv_cov_sum_sqrt * first_mean; // == 0
            auto second_mean_premult = inv_cov_sum_sqrt * second_mean;
            // spdlog::info("first_mean_premult={}", first_mean_premult);
            // spdlog::info("second_mean_premult={}", second_mean_premult);

            MatrixExpr new_cov_tangent = linalg::transpose(first_cov_premult) * second_cov_premult;
            auto new_mean_tangent =
                linalg::transpose(second_cov_premult) * first_mean_premult + // == 0
                linalg::transpose(first_cov_premult) * second_mean_premult;
            // spdlog::info("new_mean_tangent={}", new_mean_tangent);

            auto [new_mean, from_jacobian] = first_ts.from_jacobian(new_mean_tangent);
            auto&& product_ts = enoki::packet(result.tangent_space, product_i);
            product_ts.set_mean(new_mean);
            auto to_center_jacobian = product_ts.to_center_jacobian();
            J = to_center_jacobian * from_jacobian;
            // spdlog::info("new_mean={}", to_center_jacobian);
            // spdlog::info("to_center_J={}", to_center_jacobian);
            // spdlog::info("from_J={}", from_jacobian);
            // spdlog::info("J_from={}", J);

            auto new_cov = J * new_cov_tangent * linalg::transpose(J);
            // spdlog::info("new_cov={}", new_cov);

            enoki::packet(result.cov, product_i) = new_cov;

            ScalarExpr cov_det = enoki::hprod(enoki::diag(cov_sum_sqrt));
            ScalarExpr inv_cov_sqrt_det = 1.f / cov_det;
            auto standardized = sdmm::linalg::solve(cov_sum_sqrt, second_mean);
            auto squared_norm = enoki::squared_norm(standardized);
            ScalarExpr first_weight = enoki::packet(first.weight.pmf, first_i);
            ScalarS second_weight = enoki::slice(second.weight.pmf, second_i);
            ScalarExpr new_weight = 
                first_weight *
                second_weight * 
                inv_cov_sqrt_det *
                inv_jacobian *
                gaussian_normalization<ScalarS, SDMM::CovSize> *
                enoki::exp(ScalarS(-0.5) * squared_norm);

            auto dot = enoki::dot(first_ts.mean, second_ts.mean);
            enoki::packet(result.weight.pmf, product_i) = enoki::select(
                cov_det == 0, // || dot < 0,
                0,
                new_weight
            );
        }
    }
    return sdmm::prepare_vectorized(result);
}

template<
    typename SDMM,
    std::enable_if_t<
        (sdmm::embedded_t<SDMM>::Size != sdmm::tangent_t<SDMM>::Size &&
        sdmm::embedded_t<SDMM>::Size > 3),
        int
    > = 0
>
auto product(SDMM& first, SDMM& second, SDMM& result) {
    result = first;
}

template<typename SDMM, std::enable_if_t<sdmm::embedded_t<SDMM>::Size == sdmm::tangent_t<SDMM>::Size, int> = 0>
auto product(SDMM& first, SDMM& second, SDMM& result) {
    using ScalarS = typename SDMM::ScalarS;
    using EmbeddedS = sdmm::embedded_s_t<SDMM>;
    using TangentS = sdmm::tangent_s_t<SDMM>;
    using MatrixS = sdmm::matrix_s_t<SDMM>;
    size_t first_size = enoki::slices(first);
    size_t second_size = enoki::slices(second);
    size_t product_size = first_size * second_size;
    enoki::set_slices(result, product_size);
    for(size_t first_i = 0; first_i < first_size; ++first_i) {
        for(size_t second_i = 0; second_i < second_size; ++second_i) {
            ScalarS new_weight = 0;
            MatrixS first_cov = enoki::slice(first.cov, first_i);
            MatrixS second_cov = enoki::slice(second.cov, second_i);

            EmbeddedS first_mean = enoki::slice(first.tangent_space.mean, first_i);
            EmbeddedS second_mean = enoki::slice(second.tangent_space.mean, second_i);

            MatrixS cov_sum = first_cov + second_cov;
            MatrixS cov_sum_sqrt;
            typename SDMM::MaskS is_psd;
            linalg::cholesky(cov_sum, cov_sum_sqrt, is_psd);
            MatrixS inv_cov_sum_sqrt = linalg::inverse_lower_tri(cov_sum_sqrt);

            MatrixS first_cov_premult = inv_cov_sum_sqrt * first_cov;
            MatrixS second_cov_premult = inv_cov_sum_sqrt * second_cov;

            EmbeddedS first_mean_premult = inv_cov_sum_sqrt * first_mean;
            EmbeddedS second_mean_premult = inv_cov_sum_sqrt * second_mean;

            MatrixS new_cov = linalg::transpose(first_cov_premult) * second_cov_premult;
            EmbeddedS new_mean =
                linalg::transpose(second_cov_premult) * first_mean_premult +
                linalg::transpose(first_cov_premult) * second_mean_premult;

            size_t product_i = first_i * first_size + second_i;
            enoki::slice(result.cov, product_i) = new_cov;
            enoki::slice(result.tangent_space, product_i).set_mean(new_mean);
            result.weight.pmf.coeff(product_i) = 1;
        }
    }
    result.prepare();
}

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

