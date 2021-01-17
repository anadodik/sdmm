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

template<
    typename Matrix,
    typename Tangent,
    typename TangentSpace,
    typename TangentSpaceOut,
    typename MatrixOut,
    std::enable_if_t<
      !TangentSpace::HasTangentSpaceOffset, int
    > = 0
>
inline auto to_centered_tangent_space(
    const Matrix& new_cov_tangent,
    const Tangent& new_mean_tangent,
    const TangentSpace& old_tangent_space,
    TangentSpaceOut& new_tangent_space,
    MatrixOut& new_cov
) -> void {
    using TangentExpr = enoki::expr_t<Tangent>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    auto [new_mean, from_jacobian] = old_tangent_space.from_jacobian(new_mean_tangent);
    new_tangent_space.set_mean(new_mean);
    auto to_center_jacobian = new_tangent_space.to_center_jacobian();
    const MatrixExpr J = to_center_jacobian * from_jacobian;
    new_cov = J * new_cov_tangent * linalg::transpose(J);
}


template<
    typename Matrix,
    typename Tangent,
    typename TangentSpace,
    typename TangentSpaceOut,
    typename MatrixOut,
    std::enable_if_t<
      TangentSpace::HasTangentSpaceOffset, int
    > = 0
>
inline auto to_centered_tangent_space(
    const Matrix& new_cov_tangent,
    const Tangent& new_mean_tangent,
    const TangentSpace& old_tangent_space,
    TangentSpaceOut& new_tangent_space,
    MatrixOut& new_cov
) -> void {
    using TangentExpr = enoki::expr_t<Tangent>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    new_tangent_space = old_tangent_space;
    new_tangent_space.tangent_mean = old_tangent_space.tangent_mean + new_mean_tangent;
    new_cov = new_cov_tangent;
}

template<
    typename SDMM1,
    typename SDMM2,
    std::enable_if_t<
      !SDMM1::TangentSpace::IsEuclidian && !SDMM2::TangentSpace::IsEuclidian, int
    > = 0
>
auto product(SDMM1& first, SDMM2& second, SDMM1& result) {
    // TODO: implement single lobe selection from learned BSDF
    using ScalarS = typename SDMM1::ScalarS;
    using EmbeddedS = sdmm::embedded_s_t<SDMM1>;
    using TangentS = sdmm::tangent_s_t<SDMM1>;
    using MatrixS = sdmm::matrix_s_t<SDMM1>;

    using ScalarExpr = enoki::expr_t<decltype(enoki::packet(first.weight.pmf, 0))>;
    using TangentExpr = enoki::expr_t<typename decltype(enoki::packet(first.tangent_space, 0))::TangentExpr>;
    using MatrixExpr = enoki::expr_t<decltype(enoki::packet(first.cov, 0))>;

    size_t first_size = enoki::slices(first);
    size_t second_size = enoki::slices(second);
    size_t nonzero_second_size = 0;
    for(size_t second_i = 0; second_i < second_size; ++second_i) {
        if(second.weight.pmf.coeff(second_i) != 0) {
            ++nonzero_second_size;
        }
    }
    size_t product_size = first_size * nonzero_second_size;

    if(enoki::slices(result) != product_size) {
        enoki::set_slices(result, product_size);
    }

    size_t first_packets = enoki::packets(first);
    size_t product_packets = enoki::packets(result);

    bool use_jacobian_correction = false;

    for(size_t first_i = 0; first_i < first_packets; ++first_i) {
        size_t nonzero_second_i = 0;
        for(size_t second_i = 0; second_i < second_size; ++second_i) {
            if(second.weight.pmf.coeff(second_i) == 0) {
                continue;
            }
            size_t product_i = first_i * nonzero_second_size + nonzero_second_i;
            ++nonzero_second_i;

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

            auto&& product_ts = enoki::packet(result.tangent_space, product_i);
            auto&& new_cov = enoki::packet(result.cov, product_i);
            to_centered_tangent_space(
                new_cov_tangent, new_mean_tangent, first_ts, product_ts, new_cov
            );

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
                gaussian_normalization<ScalarS, SDMM1::CovSize> *
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
  typename SDMM1,
  typename SDMM2,
  std::enable_if_t<
    SDMM1::TangentSpace::IsEuclidian && SDMM2::TangentSpace::IsEuclidian, int
  > = 0
>
auto product(SDMM1& first, SDMM2& second, SDMM1& result) {
    using ScalarS = typename SDMM1::ScalarS;
    using EmbeddedS = sdmm::embedded_s_t<SDMM1>;
    using TangentS = sdmm::tangent_s_t<SDMM1>;
    using MatrixS = sdmm::matrix_s_t<SDMM1>;
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
            typename SDMM1::MaskS is_psd;
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
