#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/opt/data.h"

#define WARN_ON_FAIL(EXPR) \
    do { \
        if(!EXPR) { \
            spdlog::warn(#EXPR); \
        } \
    } while(0)

namespace sdmm {

template<typename SDMM_>
struct Stats {
    using Scalar_ = float;
    using SDMM = SDMM_;

    using Scalar = enoki::replace_scalar_t<typename SDMM::Scalar, Scalar_>;
    using Tangent = enoki::replace_scalar_t<sdmm::tangent_t<SDMM>, Scalar_>;
    using Matrix = enoki::replace_scalar_t<sdmm::matrix_t<SDMM>, Scalar_>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using TangentExpr = enoki::expr_t<Tangent>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    
    using ScalarS = enoki::scalar_t<Scalar>;

    Scalar weight;
    Tangent mean;
    Matrix cov;

    template<typename ScalarIn, typename TangentIn>
    auto plus_eq(const ScalarIn& weight_, const TangentIn& mean) -> void;

    template<typename ScalarSIn>
    auto operator*=(ScalarSIn scalar) -> Stats&;

    auto operator+=(const Stats& other) -> Stats&;

    template<typename ScalarSIn>
    auto fmadd_eq(ScalarSIn scalar, const Stats& other) -> Stats&;

    template<typename ScalarSIn>
    auto mult_eq(ScalarSIn scalar, const Stats& other) -> Stats&;

    ENOKI_STRUCT(Stats, weight, mean, cov);
};

template<typename SDMM_>
template<typename ScalarIn, typename TangentIn>
auto Stats<SDMM_>::plus_eq(const ScalarIn& weight_, const TangentIn& mean_) -> void {
    weight += weight_;
    mean += weight_ * mean_;
    cov += weight_ * sdmm::linalg::outer(mean_);
}

template<typename SDMM_>
template<typename ScalarSIn>
auto Stats<SDMM_>::operator*=(ScalarSIn scalar) -> Stats& {
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
template<typename ScalarSIn>
auto Stats<SDMM_>::fmadd_eq(ScalarSIn scalar, const Stats& other) -> Stats& {
    weight = enoki::fmadd(scalar, other.weight, weight);
    mean += scalar * other.mean;
    cov += scalar * other.cov;
    return *this;
}

template<typename SDMM_>
template<typename ScalarSIn>
auto Stats<SDMM_>::mult_eq(ScalarSIn scalar, const Stats& other) -> Stats& {
    weight = scalar * other.weight;
    mean = scalar * other.mean;
    cov = scalar * other.cov;
    return *this;
}

template<typename SDMM_>
struct EM {
    using SDMM = SDMM_;
    using SDMMScalar = typename SDMM::Scalar;
    using SDMMTangent = sdmm::tangent_t<SDMM>;

    using Scalar = typename SDMM::Scalar;
    using Tangent = sdmm::tangent_t<SDMM>;
    using Matrix = sdmm::matrix_t<SDMM>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using TangentExpr = enoki::expr_t<Tangent>;
    using MatrixExpr = enoki::expr_t<Matrix>;

    using ScalarS = enoki::scalar_t<Scalar>;

    auto set_priors(
        ScalarS weight_prior,
        ScalarS cov_prior_strength,
        const Matrix& cov_prior
    ) -> void;

    auto step(SDMM_& sdmm, Data<SDMM_>& data) -> void;

    auto compute_stats_model_parallel(
        SDMM& distribution, Data<SDMM>& data
    ) -> void;

    auto interpolate_stats() -> void;
    auto normalize_stats(Data<SDMM_>& data) -> bool;

    Stats<SDMM> stats;
    Stats<SDMM> batch_stats;
    Stats<SDMM> stats_normalized;
    Stats<SDMM> updated_model;
    typename SDMM::ScalarS total_weight;

    SDMMScalar posteriors;
    SDMMTangent tangent;

    int iterations_run = 0;
    ScalarS learning_rate = 0.2;
    ScalarS alpha = 0.5;

    ScalarS weight_prior;
    ScalarS cov_prior_strength;
    Matrix cov_prior;
    Matrix depth_prior;

    ENOKI_STRUCT(
        EM,
        stats,
        batch_stats,
        stats_normalized,
        updated_model,
        total_weight,

        posteriors,
        tangent,

        iterations_run,

        weight_prior,
        cov_prior_strength,
        cov_prior,
        depth_prior
    );
};

template<typename SDMM_>
auto update_model(SDMM_& distribution, EM<SDMM_>& em) -> void;

template<typename SDMM_>
auto EM<SDMM_>::set_priors(
    ScalarS weight_prior_, ScalarS cov_prior_strength_, const Matrix& cov_prior_
) -> void {
    weight_prior = weight_prior_;
    cov_prior_strength = cov_prior_strength_;
    cov_prior = cov_prior_;
}

template<typename SDMM_>
auto em_step(SDMM_& distribution, EM<SDMM_>& em, Data<SDMM_>& data) -> void {
    em.compute_stats_model_parallel(distribution, data);
    spdlog::info("em.stats_normalized.weight={}", em.stats_normalized.weight);
    em.interpolate_stats();
    spdlog::info("em.stats_normalized.weight={}", em.stats_normalized.weight);
    bool success_normalized = em.normalize_stats(data);
    if(!success_normalized) {
        return;
    }
    assert(success_normalized);
    spdlog::info("em.stats_normalized.weight={}", em.stats_normalized.weight);
    if(enoki::packets(em) != enoki::packets(distribution)) {
        spdlog::warn("Different number of packets!");
    }

    enoki::vectorize_safe(
        VECTORIZE_WRAP(update_model), distribution, em
    );
    spdlog::info("distribution.mean={}", distribution.tangent_space.mean);
    bool success_prepare = sdmm::prepare_vectorized(distribution);
    assert(success_prepare);
    ++em.iterations_run;
}

template<typename SDMM_>
auto EM<SDMM_>::step(SDMM_& distribution, Data<SDMM_>& data) -> void {
    compute_stats_model_parallel(distribution, data);
    interpolate_stats();
    bool success_normalized = normalize_stats(data);
    if(!success_normalized) {
        return;
    }
    assert(success_normalized);
    update_model(distribution, *this);
    bool success_prepare = sdmm::prepare_vectorized(distribution);
    assert(success_prepare);
    ++iterations_run;
}

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
        if(std::isnan(data_slice.weight) || data_slice.weight < 1e-8) {
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
        auto posterior_sum = enoki::hsum(posteriors); 
        if(data_slice.heuristic_pdf != -1) {
            posterior_sum = 0.5 * posterior_sum + 0.5 * data_slice.heuristic_pdf;
        }
        if(posterior_sum == 0) {
            continue;
        }
        auto rcp_posterior = 1 / posterior_sum;
        if(data_slice.heuristic_pdf != -1) {
            enoki::vectorize(
                [](auto&& value) { value *= ScalarS(0.5); },
                posteriors
            );
        }
        enoki::vectorize(
            [rcp_posterior](auto&& value) { value *= rcp_posterior; },
            posteriors
        );
        // spdlog::info("slice_{} posteriors={}", slice_i, posteriors);
        // spdlog::info("slice_{} posterior_sum={}", slice_i, enoki::hsum(posteriors));
        // spdlog::info("slice_{} weight={}", slice_i, data_slice.weight);

        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(plus_eq),
            batch_stats,
            
            data_slice.weight * posteriors,
            tangent
        );
    }
    // spdlog::info("batch_stats={}", batch_stats.weight);
}

template<typename SDMM_>
auto EM<SDMM_>::interpolate_stats() -> void {
    ScalarS eta = enoki::pow(learning_rate * ScalarS(iterations_run) + 1.f, -alpha);
    // spdlog::info("eta={}", eta);
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
    ScalarS eta = enoki::pow(learning_rate * ScalarS(iterations_run) + 1.f, -alpha);
    ScalarS weight_sum = data.sum_weights();
    total_weight = (1 - eta) * total_weight + eta * weight_sum;
    ScalarS rcp_weight_sum = 1 / total_weight;
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
auto update_model(SDMM_& distribution, EM<SDMM_>& em) -> void {
    spdlog::info("em.stats_normalized.weight={}", em.stats_normalized.weight);
    using ScalarExpr = typename Stats<SDMM_>::ScalarExpr;
    using MatrixExpr = typename Stats<SDMM_>::MatrixExpr;
    ScalarExpr weight_prior_decay = 1.0 / enoki::pow(3.0, enoki::min(30, em.iterations_run));
    ScalarExpr cov_prior_strength_decay = 1.0 / enoki::pow(2.0, enoki::min(30, em.iterations_run));

    ScalarExpr weight_prior_decayed = em.weight_prior * weight_prior_decay;
    ScalarExpr cov_prior_strength_decayed = em.cov_prior_strength * cov_prior_strength_decay;
    MatrixExpr cov_prior_decayed = em.cov_prior * cov_prior_strength_decayed;

    // spdlog::info("weight_prior_decay={}", weight_prior_decay);
    // spdlog::info("cov_prior_strength_decay={}", cov_prior_strength_decay);

    ScalarExpr rcp_weight = 1.f / em.stats_normalized.weight;
    ScalarExpr rcp_cov_weight =
        1.f / (cov_prior_strength_decayed + em.stats_normalized.weight);

    // Following should always be true:
    WARN_ON_FAIL(enoki::all(enoki::isfinite(em.stats_normalized.weight)));
    WARN_ON_FAIL(enoki::all(enoki::isfinite(rcp_cov_weight)));
    WARN_ON_FAIL(enoki::all(enoki::isfinite(distribution.weight.pmf)));
    WARN_ON_FAIL(enoki::any(distribution.weight.pmf > 0));

    auto non_zero_weights = distribution.weight.pmf > 0;
    auto finite_weights = enoki::isfinite(rcp_weight);

    if(!enoki::all(finite_weights)) {
        WARN_ON_FAIL(enoki::all(finite_weights));
        // return;
    }

    // TODO: need to check whether number * rcp_cov_weight overflows or is nan!
    em.updated_model.weight = enoki::select(
        non_zero_weights,
        em.stats_normalized.weight + weight_prior_decayed,
        0
    );

    em.updated_model.mean = enoki::select(
        non_zero_weights && finite_weights,
        em.stats_normalized.mean * rcp_weight,
        0
    );

    MatrixExpr mean_subtraction = 
            sdmm::linalg::outer(em.stats_normalized.mean) * rcp_weight;
    MatrixExpr cov_unnormalized = 
        em.stats_normalized.cov - mean_subtraction + cov_prior_decayed;
    em.updated_model.cov = enoki::select(
        non_zero_weights && finite_weights,
        cov_unnormalized * rcp_cov_weight + em.depth_prior,
        distribution.cov
    );
    distribution.weight.pmf = em.updated_model.weight;
    typename SDMM_::ScalarExpr inv_jacobian;
    distribution.tangent_space.set_mean(
        distribution.tangent_space.from(em.updated_model.mean, inv_jacobian)
    );
    distribution.cov = em.updated_model.cov;

    em.stats_normalized.cov -= linalg::outer(em.stats_normalized.mean) * rcp_weight;
    em.stats.cov = enoki::select(
        non_zero_weights && finite_weights,
        em.stats_normalized.cov * em.total_weight,
        em.stats.cov
    );
    em.stats.mean = enoki::select(
        non_zero_weights && finite_weights,
        0,
        em.stats.mean
    );
}

}

ENOKI_STRUCT_SUPPORT(sdmm::Stats, weight, mean, cov);
ENOKI_STRUCT_SUPPORT(
    sdmm::EM,
    stats,
    batch_stats,
    stats_normalized,
    updated_model,
    total_weight,

    posteriors,
    tangent,

    iterations_run,

    weight_prior,
    cov_prior_strength,
    cov_prior,
    depth_prior
);
