#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/opt/data.h"

namespace sdmm {

template<typename SDMM_>
struct Stats {
    using SDMM = SDMM_;
    using Scalar = typename SDMM::Scalar;
    using Tangent = sdmm::tangent_t<SDMM>;
    using Matrix = sdmm::matrix_t<SDMM>;
    
    using ScalarS = enoki::scalar_t<Scalar>;

    Scalar weight;
    Tangent mean;
    Matrix cov;

    auto plus_eq(const Scalar& weight_, const Tangent& mean) -> void;

    auto operator*=(ScalarS scalar) -> Stats&;
    auto operator+=(const Stats& other) -> Stats&;
    auto fmadd_eq(ScalarS scalar, const Stats& other) -> Stats&;
    auto mult_eq(ScalarS scalar, const Stats& other) -> Stats&;

    ENOKI_STRUCT(Stats, weight, mean, cov);
};

template<typename SDMM_>
auto Stats<SDMM_>::plus_eq(const Scalar& weight_, const Tangent& mean_) -> void {
    weight += weight_;
    mean += weight_ * mean_;
    cov += weight_ * sdmm::linalg::outer(mean_);
}

template<typename SDMM_>
auto Stats<SDMM_>::operator*=(ScalarS scalar) -> Stats& {
    weight *= scalar;
    mean *= scalar;
    cov *= scalar;
    return *this;
}

template<typename SDMM_>
auto Stats<SDMM_>::operator+=(const Stats& other) -> Stats& {
    weight += other.weight;
    mean += other.mean;
    cov += other.cov;
    return *this;
}

template<typename SDMM_>
auto Stats<SDMM_>::fmadd_eq(ScalarS scalar, const Stats& other) -> Stats& {
    weight = enoki::fmadd(scalar, other.weight, weight);
    mean += scalar * other.mean;
    cov += scalar * other.cov;
    return *this;
}

template<typename SDMM_>
auto Stats<SDMM_>::mult_eq(ScalarS scalar, const Stats& other) -> Stats& {
    weight = scalar * other.weight;
    mean = scalar * other.mean;
    cov = scalar * other.cov;
    return *this;
}

template<typename SDMM_>
struct EM {
    using SDMM = SDMM_;
    using Scalar = typename SDMM::Scalar;
    using Tangent = sdmm::tangent_t<SDMM>;
    using Matrix = sdmm::matrix_t<SDMM>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using MatrixExpr = enoki::expr_t<Matrix>;

    using ScalarS = enoki::scalar_t<Scalar>;

    auto compute_stats_model_parallel(
        SDMM& distribution, Data<SDMM>& data
    ) -> void;

    auto interpolate_stats() -> void;
    auto normalize_stats(Data<SDMM_>& data) -> bool;

    auto update_model(SDMM& distribution) -> void;

    auto set_priors(
        ScalarS weight_prior,
        ScalarS cov_prior_strength,
        const Matrix& cov_prior
    ) -> void;

    Stats<SDMM> stats;
    Stats<SDMM> batch_stats;
    Stats<SDMM> stats_normalized;
    Stats<SDMM> updated_model;

    Scalar posteriors;
    Tangent tangent;

    int iterations_run = 0;
    ScalarS learning_rate = 0.2;
    ScalarS alpha = 0.5;

    ScalarS weight_prior;
    ScalarS cov_prior_strength;
    Matrix cov_prior;

    ENOKI_STRUCT(
        EM,
        stats,
        batch_stats,
        stats_normalized,
        updated_model,

        posteriors,
        tangent,

        iterations_run,
        learning_rate,
        alpha,

        weight_prior,
        cov_prior_strength,
        cov_prior
    );
};

template<typename SDMM_>
auto EM<SDMM_>::compute_stats_model_parallel(
    SDMM& distribution, Data<SDMM_>& data
) -> void {
    batch_stats = enoki::zero<Stats<SDMM>>(enoki::slices(batch_stats));
    // enoki::vectorize(
    //     [](auto&& stats) { stats = enoki::zero<enoki::expr_t<decltype(stats)>>(); },
    //     batch_stats
    // );
    for(size_t slice_i = 0; slice_i < data.size; ++slice_i) {
        auto data_slice = enoki::slice(data, slice_i);
        if(std::isnan(data_slice.weight) || data_slice.weight == 0) {
            continue;
        }
        // spdlog::info("slice_{} weight={}", slice_i, data_slice.weight);
        // spdlog::info("slice_{} point={}", slice_i, data_slice.point);
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(pdf_gaussian),
            distribution,
            data_slice.point,
            posteriors,
            tangent
        );
        // spdlog::info("tangent={}", tangent);
        // spdlog::info("slice_{} pdf={}", slice_i, posteriors);
        enoki::vectorize(
            [](auto&& value, auto&& weight) { value *= weight; },
            posteriors,
            distribution.weight.pmf
        );
        // spdlog::info("slice_{} pmf={}", slice_i, distribution.weight.pmf);
        // spdlog::info("slice_{} posterior={}", slice_i, posteriors);
        auto posterior_sum = enoki::hsum(posteriors); 
        if(data_slice.heuristic_pdf > 0) {
            posterior_sum = 0.5 * posterior_sum + 0.5 * data_slice.heuristic_pdf;
        }
        if(posterior_sum == 0) {
            continue;
        }
        auto rcp_posterior = 1 / posterior_sum;
        enoki::vectorize(
            [rcp_posterior](auto&& value) { value *= rcp_posterior; },
            posteriors
        );
        spdlog::info("slice_{} posteriors={}", slice_i, posteriors);
        // spdlog::info("slice_{} posterior_sum={}", slice_i, enoki::hsum(posteriors));
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(plus_eq),
            batch_stats,
            
            data_slice.weight * posteriors,
            tangent
        );
    }
}

template<typename SDMM_>
auto EM<SDMM_>::interpolate_stats() -> void {
    ScalarS eta = enoki::pow(learning_rate * iterations_run + 1, -alpha);
    enoki::vectorize_safe(
        [eta](auto&& stats, auto&& batch_stats) {
            stats *= 1 - eta;
            stats.fmadd_eq(eta, batch_stats);
        },
        stats,
        batch_stats
    );
}
template<typename SDMM_>
[[nodiscard]] auto EM<SDMM_>::normalize_stats(Data<SDMM_>& data) -> bool {
    enoki::vectorize_safe(
        VECTORIZE_WRAP_MEMBER(remove_non_finite), data
    );
    ScalarS weight_sum = enoki::hsum(data.weight);
    ScalarS rcp_weight_sum = 1 / weight_sum;
    if(!enoki::isfinite(rcp_weight_sum)) {
        spdlog::warn("weight_sum == 0");
        return false;
    }
    enoki::vectorize_safe(
        [rcp_weight_sum](auto&& stats, auto&& stats_normalized) {
            stats_normalized.mult_eq(rcp_weight_sum, stats);
        },
        stats,
        stats_normalized
    );
    return true;
}

template<typename SDMM_>
auto EM<SDMM_>::set_priors(
    ScalarS weight_prior_, ScalarS cov_prior_strength_, const Matrix& cov_prior_
) -> void {
    weight_prior = weight_prior_;
    cov_prior_strength = cov_prior_strength_;
    cov_prior = cov_prior_;
}

template<typename SDMM_>
auto EM<SDMM_>::update_model(SDMM& distribution) -> void {
    ScalarS weight_prior_decay = ScalarS(1) / enoki::pow(3, iterations_run);
    ScalarS weight_prior_decayed = weight_prior * weight_prior_decay;

    ScalarS cov_prior_strength_decay = ScalarS(1) / enoki::pow(2, iterations_run);
    ScalarS cov_prior_strength_decayed = cov_prior_strength * cov_prior_strength_decay;
    MatrixExpr cov_prior_decayed = cov_prior * cov_prior_strength_decayed;

    ScalarExpr rcp_weight = 1 / stats_normalized.weight;
    ScalarExpr rcp_cov_weight =
        1 / (cov_prior_strength_decayed + stats_normalized.weight); // TODO: 0.05 factor

    // Following should always be true:
    assert(enoki::isfinite(stats_normalized.weight) == true);
    assert(enoki::isfinite(rcp_cov_weight) == true);
    assert(enoki::all(distribution.weight.pmf > 0));

    auto non_zero_weights = distribution.weight.pmf > 0;
    auto finite_weights = enoki::isfinite(rcp_weight);

    // TODO: need to check whether number * rcp_cov_weight overflows or is nan!
    updated_model.weight = enoki::select(
        non_zero_weights,
        stats_normalized.weight + weight_prior_decayed,
        ScalarS(0)
    );

    updated_model.mean = enoki::select(
        non_zero_weights && finite_weights,
        stats_normalized.mean * rcp_weight,
        0
    );

    updated_model.cov = enoki::select(
        non_zero_weights && finite_weights,
        (
            stats_normalized.cov -
            sdmm::linalg::outer(stats_normalized.mean) *
            rcp_weight +
            cov_prior_decayed
        ) * rcp_cov_weight,
        distribution.cov
    );

    distribution.weight.pmf = updated_model.weight;
    ScalarExpr inv_jacobian;
    distribution.tangent_space.set_mean(
        distribution.tangent_space.from(updated_model.mean, inv_jacobian)
    );
    distribution.cov = updated_model.cov;
}

}

ENOKI_STRUCT_SUPPORT(sdmm::Stats, weight, mean, cov);
ENOKI_STRUCT_SUPPORT(
    sdmm::EM,
    stats,
    batch_stats,
    stats_normalized,
    updated_model,

    posteriors,
    tangent,

    iterations_run,
    learning_rate,
    alpha,

    weight_prior,
    cov_prior_strength,
    cov_prior
);
