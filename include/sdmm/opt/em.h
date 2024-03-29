#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/opt/data.h"


#define WARN_ON_FAIL(EXPR, OUT)             \
    do {                                    \
        if (false && !EXPR) {                        \
            spdlog::warn(#EXPR "={}", OUT); \
        }                                   \
    } while (0)

namespace sdmm {

template <typename SDMM_>
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

    template <typename ScalarIn, typename TangentIn>
    auto plus_eq(const ScalarIn& weight_, const TangentIn& mean) -> void;

    template <typename ScalarSIn>
    auto operator*=(ScalarSIn scalar) -> Stats&;

    auto operator+=(const Stats& other) -> Stats&;

    template <typename ScalarSIn>
    auto fmadd_eq(ScalarSIn scalar, const Stats& other) -> Stats&;

    template <typename ScalarSIn>
    auto mult_eq(ScalarSIn scalar, const Stats& other) -> Stats&;

    ENOKI_STRUCT(Stats, weight, mean, cov);
};

template <typename SDMM_>
template <typename ScalarIn, typename TangentIn>
auto Stats<SDMM_>::plus_eq(const ScalarIn& weight_, const TangentIn& mean_)
    -> void {
    weight += weight_;
    mean += weight_ * mean_;
    cov += weight_ * sdmm::linalg::outer(mean_);
}

template <typename SDMM_>
template <typename ScalarSIn>
auto Stats<SDMM_>::operator*=(ScalarSIn scalar) -> Stats& {
    weight *= scalar;
    mean *= scalar;
    cov *= scalar;
    return *this;
}

template <typename SDMM_>
auto Stats<SDMM_>::operator+=(const Stats& other) -> Stats& {
    weight += other.weight;
    mean += other.mean;
    cov += other.cov;
    return *this;
}

template <typename SDMM_>
template <typename ScalarSIn>
auto Stats<SDMM_>::fmadd_eq(ScalarSIn scalar, const Stats& other) -> Stats& {
    weight = enoki::fmadd(scalar, other.weight, weight);
    mean += scalar * other.mean;
    cov += scalar * other.cov;
    return *this;
}

template <typename SDMM_>
template <typename ScalarSIn>
auto Stats<SDMM_>::mult_eq(ScalarSIn scalar, const Stats& other) -> Stats& {
    weight = scalar * other.weight;
    mean = scalar * other.mean;
    cov = scalar * other.cov;
    return *this;
}

template <typename SDMM_>
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
        const Matrix& cov_prior) -> void;

    auto compute_stats_model_parallel(SDMM& distribution, Data<SDMM>& data)
        -> void;

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
    int samples_seen = 0;
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
        samples_seen,

        weight_prior,
        cov_prior_strength,
        cov_prior,
        depth_prior);
};

template <typename SDMM_>
auto update_model(SDMM_& distribution, EM<SDMM_>& em) -> void;

template <typename SDMM_>
auto EM<SDMM_>::set_priors(
    ScalarS weight_prior_,
    ScalarS cov_prior_strength_,
    const Matrix& cov_prior_) -> void {
    weight_prior = weight_prior_;
    cov_prior_strength = cov_prior_strength_;
    cov_prior = cov_prior_;
}

template <typename SDMM_>
auto e_step(SDMM_& distribution, EM<SDMM_>& em, Data<SDMM_>& data) -> void {
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(distribution.weight.pmf)),
        distribution.weight.pmf);
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(em.stats_normalized.weight)),
        em.stats_normalized.weight);
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(em.batch_stats.weight)),
        em.batch_stats.weight);
    WARN_ON_FAIL(enoki::all(enoki::isfinite(em.stats.weight)), em.stats.weight);

    em.compute_stats_model_parallel(distribution, data);
    // spdlog::info("em.stats_normalized.weight={}",
    // em.stats_normalized.weight);
    em.interpolate_stats();
    // spdlog::info("em.stats_normalized.weight={}",
    // em.stats_normalized.weight);
    bool success_normalized = em.normalize_stats(data);
    if (!success_normalized) {
        return;
    }
    assert(success_normalized);
    // spdlog::info("em.stats_normalized.weight={}",
    // em.stats_normalized.weight);
    if (enoki::packets(em) != enoki::packets(distribution)) {
        spdlog::warn("Different number of packets!");
    }
}

template <typename SDMM_>
auto m_step(SDMM_& distribution, EM<SDMM_>& em) -> void {
    enoki::vectorize_safe(VECTORIZE_WRAP(update_model), distribution, em);
    // spdlog::info("distribution.mean={}", distribution.tangent_space.mean);
    bool success_prepare = sdmm::prepare_vectorized(distribution);
    assert(success_prepare);
    ++em.iterations_run;
}

template <typename SDMM_>
auto em_step(SDMM_& distribution, EM<SDMM_>& em, Data<SDMM_>& data) -> void {
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(distribution.weight.pmf)),
        distribution.weight.pmf);
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(em.stats_normalized.weight)),
        em.stats_normalized.weight);
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(em.batch_stats.weight)),
        em.batch_stats.weight);
    WARN_ON_FAIL(enoki::all(enoki::isfinite(em.stats.weight)), em.stats.weight);

    em.compute_stats_model_parallel(distribution, data);
    // spdlog::info("em.stats_normalized.weight={}",
    // em.stats_normalized.weight);
    em.interpolate_stats();
    // spdlog::info("em.stats_normalized.weight={}",
    // em.stats_normalized.weight);
    bool success_normalized = em.normalize_stats(data);
    if (!success_normalized) {
        return;
    }
    assert(success_normalized);
    // spdlog::info("em.stats_normalized.weight={}",
    // em.stats_normalized.weight);
    if (enoki::packets(em) != enoki::packets(distribution)) {
        spdlog::warn("Different number of packets!");
    }

    enoki::vectorize_safe(VECTORIZE_WRAP(update_model), distribution, em);
    // spdlog::info("distribution.mean={}", distribution.tangent_space.mean);
    bool success_prepare = sdmm::prepare_vectorized(distribution);
    assert(success_prepare);
    ++em.iterations_run;
}

template <typename SDMM_>
auto EM<SDMM_>::compute_stats_model_parallel(
    SDMM& distribution,
    Data<SDMM_>& data) -> void {
    batch_stats = enoki::zero<Stats<SDMM>>(enoki::slices(batch_stats));
    // enoki::vectorize(
    //     [](auto&& stats) { stats =
    //     enoki::zero<enoki::expr_t<decltype(stats)>>(); }, batch_stats
    // );
    samples_seen += data.size;
    for (size_t slice_i = 0; slice_i < data.size; ++slice_i) {
        auto data_slice = enoki::slice(data, slice_i);
        if (!sdmm::is_valid_sample(data_slice.weight)) {
            continue;
        }

        sdmm::embedded_s_t<SDMM> point = data_slice.point;
        // auto length = enoki::norm(enoki::tail<3>(point));
        // if(enoki::any(enoki::abs(length - 1) >= 1e-5)) {
        //     std::cerr << fmt::format("length=({}, {})\n", data_slice.point,
        //     data_slice.weight);
        // }
        // spdlog::info("slice_{} weight={}", slice_i, data_slice.weight);
        // spdlog::info("slice_{} point={}", slice_i, data_slice.point);
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(posterior),
            distribution,
            data_slice.point,
            posteriors,
            tangent);
        
        // spdlog::info("tangent={}", tangent);
        // spdlog::info("slice_{} pdf={}", slice_i, posteriors);
        // spdlog::info("slice_{} pmf={}", slice_i, distribution.weight.pmf);
        auto posterior_sum = enoki::hsum(posteriors);
        if (posterior_sum == 0 || !std::isfinite(1.f / posterior_sum)) {
            --samples_seen;
            continue;
        }

        auto rcp_posterior = 1.f / posterior_sum;
        enoki::vectorize(
            [rcp_posterior](auto&& value) { value *= rcp_posterior; },
            posteriors);
        // spdlog::info("slice_{} posteriors={}", slice_i, posteriors);
        // spdlog::info("slice_{} posterior_sum={}", slice_i,
        // enoki::hsum(posteriors)); spdlog::info("slice_{} weight={}", slice_i,
        // data_slice.weight);

        // spdlog::info("batch_stats={}, posteriors={}, tangent={}",
        // batch_stats.weight, posteriors, tangent);
        enoki::vectorize(
            VECTORIZE_WRAP_MEMBER(plus_eq),
            batch_stats,

            data_slice.weight * posteriors,
            tangent);
    }
    // spdlog::info("batch_stats={}", batch_stats.weight);
}

template <typename SDMM_>
auto EM<SDMM_>::interpolate_stats() -> void {
    ScalarS eta =
        enoki::pow(learning_rate * ScalarS(iterations_run) + 1.f, -alpha);
    // spdlog::info("eta={}", eta);
    enoki::vectorize_safe(
        [eta](auto&& stats, auto&& batch_stats) {
            stats *= 1 - eta;
            stats.fmadd_eq(eta, batch_stats);
        },
        stats,
        batch_stats);
}

template <typename SDMM_>
[[nodiscard]] auto EM<SDMM_>::normalize_stats(Data<SDMM_>& data) -> bool {
    ScalarS eta =
        enoki::pow(learning_rate * ScalarS(iterations_run) + 1.f, -alpha);
    ScalarS weight_sum = data.sum_weights();
    total_weight = (1 - eta) * total_weight + eta * weight_sum;
    ScalarS rcp_weight_sum = 1 / total_weight;
    if (!enoki::isfinite(rcp_weight_sum)) {
        spdlog::warn("!std::isfinite(weight_sum)");
        return false;
    } else if (total_weight == 0) {
        spdlog::warn("total_weight = 0");
        return false;
    }
    enoki::vectorize_safe(
        [rcp_weight_sum](auto&& stats, auto&& stats_normalized) {
            stats_normalized.mult_eq(rcp_weight_sum, stats);
        },
        stats,
        stats_normalized);
    return true;
}

template <typename SDMM_>
auto update_model(SDMM_& distribution, EM<SDMM_>& em) -> void {
    // spdlog::info("em.stats_normalized.weight={}",
    // em.stats_normalized.weight);
    using ScalarS = typename Stats<SDMM_>::ScalarS;
    using ScalarExpr = typename Stats<SDMM_>::ScalarExpr;
    using MatrixExpr = typename Stats<SDMM_>::MatrixExpr;

    ScalarS weight_prior_decay =
        // 1.0 / ((float) em.samples_seen);
        ScalarS(1.0 / enoki::pow(3.0, enoki::min(30, em.iterations_run)));
    ScalarS cov_prior_strength_decay =
        // 1.0 / ((float) em.samples_seen);
        ScalarS(1.0 / enoki::pow(2.0, enoki::min(30, em.iterations_run)));

    ScalarS weight_prior_decayed = em.weight_prior * weight_prior_decay;
    ScalarS cov_prior_strength_decayed =
        em.cov_prior_strength * cov_prior_strength_decay;
    MatrixExpr cov_prior_decayed = em.cov_prior * cov_prior_strength_decayed;

    // spdlog::info("weight_prior_decay={}", weight_prior_decay);
    // spdlog::info("cov_prior_strength_decay={}", cov_prior_strength_decay);

    ScalarExpr rcp_weight = 1.f / em.stats_normalized.weight;
    ScalarExpr rcp_cov_weight =
        1.f / (cov_prior_strength_decayed + em.stats_normalized.weight);

    auto non_zero_weights = distribution.weight.pmf > 0;
    auto finite_rcp_weights = enoki::isfinite(rcp_weight);

    // Following should always be true:
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(em.stats_normalized.weight)),
        em.stats_normalized.weight);
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(em.batch_stats.weight)),
        em.batch_stats.weight);
    WARN_ON_FAIL(enoki::all(enoki::isfinite(em.stats.weight)), em.stats.weight);
    WARN_ON_FAIL((em.total_weight > 0), em.total_weight);
    WARN_ON_FAIL(enoki::all(enoki::isfinite(rcp_cov_weight)), rcp_cov_weight);
    WARN_ON_FAIL(
        enoki::all(enoki::isfinite(distribution.weight.pmf)),
        distribution.weight.pmf);
    WARN_ON_FAIL(
        enoki::any(distribution.weight.pmf > 0), distribution.weight.pmf);
    WARN_ON_FAIL(enoki::any(non_zero_weights), non_zero_weights);
    WARN_ON_FAIL(enoki::all(finite_rcp_weights), rcp_weight);

    // TODO: need to check whether number * rcp_cov_weight overflows or is nan!
    em.updated_model.weight = enoki::select(
        non_zero_weights,
        enoki::select(
            finite_rcp_weights,
            em.stats_normalized.weight + weight_prior_decayed,
            weight_prior_decayed),
        0);

    // if (!enoki::all(enoki::isfinite(em.updated_model.weight))) {
    //     std::cout << fmt::format(
    //         "em.updated_model.weight={}\n"

    //         "distribution=\n"

    //         "weight={}\n"
    //         // "to={}\n"
    //         "mean={}\n"
    //         "cond_cov={}\n"
    //         "cov_sqrt={}\n"
    //         "",
    //         em.updated_model.weight,

    //         distribution.weight.pmf,
    //         // distribution.tangent_space.coordinate_system.to,
    //         distribution.tangent_space.mean,
    //         distribution.cov,
    //         distribution.cov_sqrt);
    // }

    em.updated_model.mean = enoki::select(
        non_zero_weights && finite_rcp_weights,
        em.stats_normalized.mean * rcp_weight,
        0);

    MatrixExpr mean_subtraction =
        sdmm::linalg::outer(em.stats_normalized.mean) * rcp_weight;
    // Debug<decltype(blub), decltype(rcp_weight),
    // decltype(sdmm::linalg::outer(em.stats_normalized.mean))> debug;

    MatrixExpr cov_unnormalized =
        em.stats_normalized.cov - mean_subtraction + cov_prior_decayed;

    MatrixExpr cov_normalized = cov_unnormalized * rcp_cov_weight;
    // auto finite_mat = [](auto&& mat) {
    //     for (size_t i = 0; i < std::decay_t<decltype(mat)>::Rows; ++i) {
    //         for (size_t j = 0; j < std::decay_t<decltype(mat)>::Cols; ++j) {
    //             bool isfinite = enoki::all(enoki::isfinite(mat(i, j)));
    //             if (!isfinite) {
    //                 return false;
    //             }
    //         }
    //     }
    //     return true;
    // };

    // if (!finite_mat(cov_normalized)) {
    //     spdlog::warn(
    //         "!isfinite(cov)={}=\n{}\n*{}\n",
    //         cov_normalized,
    //         cov_unnormalized,
    //         rcp_cov_weight);
    // }
    em.updated_model.cov = enoki::select(
        non_zero_weights && finite_rcp_weights,
        cov_normalized + em.depth_prior,
        distribution.cov);
    distribution.weight.pmf = em.updated_model.weight;
    // typename SDMM_::ScalarExpr inv_jacobian;
    auto [new_embedded_mean, from_jacobian] =
        distribution.tangent_space.from_jacobian(em.updated_model.mean);
    distribution.tangent_space.set_mean(new_embedded_mean);
    auto to_jacobian = distribution.tangent_space.to_center_jacobian();
    auto jacobian = to_jacobian * from_jacobian;
    distribution.cov =
        jacobian * em.updated_model.cov * linalg::transpose(jacobian);

    em.stats_normalized.cov -=
        linalg::outer(em.stats_normalized.mean) * rcp_weight;
    em.stats_normalized.cov =
        jacobian * em.stats_normalized.cov * linalg::transpose(jacobian);
    em.stats.cov = enoki::select(
        non_zero_weights && finite_rcp_weights,
        em.stats_normalized.cov * em.total_weight,
        em.stats.cov);
    em.stats.mean =
        enoki::select(non_zero_weights && finite_rcp_weights, 0, em.stats.mean);
}

} // namespace sdmm

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
    samples_seen,

    weight_prior,
    cov_prior_strength,
    cov_prior,
    depth_prior);
