#pragma once

#include "sdmm/distributions/sdmm.h"

namespace sdmm {

template <typename Joint, typename Marginal>
inline auto create_marginal(const Joint& joint, Marginal& marginal) -> void {
    static constexpr size_t MarginalSize = Marginal::CovSize;
    using MarginalTangentSpace = tangent_space_t<Marginal>;
    marginal.weight.pmf = joint.weight.pmf;
    marginal.weight.cdf = joint.weight.cdf;

    typename MarginalTangentSpace::EmbeddedExpr mean;
    for (size_t r = 0; r < MarginalSize; ++r) {
        mean.coeff(r) = joint.tangent_space.mean.coeff(r);
        for (size_t c = 0; c < MarginalSize; ++c) {
            marginal.cov(r, c) = joint.cov(r, c);
        }
    }
    for (size_t r = 0; r < MarginalTangentSpace::Embedded::Size; ++r) {
        mean.coeff(r) = joint.tangent_space.mean.coeff(r);
    }
    marginal.tangent_space.set_mean(mean);
    marginal.prepare_cov();
}

// TODO: rename to SDMMDecomposition?
template <typename Joint_, typename Marginal_, typename Conditional_>
struct SDMMConditioner {
    using Joint = Joint_;
    using Marginal = Marginal_;
    using Conditional = Conditional_;

    static constexpr size_t JointSize = Joint::CovSize;
    static constexpr size_t MarginalSize = Marginal::CovSize;
    static constexpr size_t ConditionalSize = Conditional::CovSize;

    using JointTangentSpace = tangent_space_t<Joint>;
    using MarginalTangentSpace = tangent_space_t<Marginal>;
    using ConditionalTangentSpace = tangent_space_t<Conditional>;

    using ScalarExpr = enoki::expr_t<enoki::value_t<embedded_t<Joint>>>;
    using JointMatrix = matrix_t<Joint>;
    using MarginalEmbeddedS = embedded_s_t<Marginal>;
    using MeanTransformMatrix = typename JointMatrix::
        template ReplaceSize<ConditionalSize, MarginalSize>;
    using MarginalEmbedded = typename Marginal::Embedded;

    auto prepare_vectorized(const Joint& joint) -> void;

    template <
        typename C = Conditional_,
        std::enable_if_t<C::TangentSpace::HasTangentSpaceOffset, int> = 0>
    auto create_conditional_vectorized(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void;

    template <
        typename C = Conditional_,
        std::enable_if_t<!C::TangentSpace::HasTangentSpaceOffset, int> = 0>
    auto create_conditional_vectorized(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void;

    auto create_conditional_weights_vectorized(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void;
    auto create_conditional_means_and_covs(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void;

    Marginal marginal;
    Conditional conditional;

    MeanTransformMatrix mean_transform;
    ConditionalTangentSpace tangent_space;
    MarginalEmbedded marginal_tangents;

    ENOKI_STRUCT(
        SDMMConditioner,
        marginal,
        conditional,
        mean_transform,
        tangent_space,
        marginal_tangents);
};

template <typename Conditioner>
inline auto prepare(
    // cannot declare joint as const because enoki complains
    Conditioner& conditioner,
    typename Conditioner::Joint& joint) -> void {
    enoki::vectorize_safe(
        VECTORIZE_WRAP_MEMBER(prepare_vectorized),
        std::forward<Conditioner>(conditioner),
        joint);
    bool cdf_success = conditioner.marginal.weight.prepare();
    assert(cdf_success);
}

template <typename Conditioner>
inline auto create_conditional(
    // cannot declare point as const because enoki complains
    Conditioner& conditioner,
    typename Conditioner::MarginalEmbeddedS& point,
    typename Conditioner::Conditional& out) -> bool {
    enoki::vectorize_safe(
        VECTORIZE_WRAP_MEMBER(create_conditional_vectorized),
        conditioner,
        point,
        out);
    bool cdf_success = out.weight.prepare();
    return cdf_success;
}

template <typename Conditioner>
inline auto create_conditional_static(
    // cannot declare point as const because enoki complains
    Conditioner& conditioner,
    typename Conditioner::MarginalEmbeddedS& point,
    typename Conditioner::Conditional& out) -> bool {
    create_conditional_vectorized(conditioner, point, out);
    bool cdf_success = out.weight.prepare();
    return cdf_success;
}

template <typename Conditioner>
inline auto create_conditional_pruned(
    // cannot declare point as const because enoki complains
    Conditioner& conditioner,
    typename Conditioner::MarginalEmbeddedS& point,
    typename Conditioner::Conditional& out,
    size_t max_components,
    size_t preserve_idx = -1) -> bool {
    using ScalarS = typename Conditioner::Joint::ScalarS;

    enoki::vectorize_safe(
        VECTORIZE_WRAP_MEMBER(create_conditional_vectorized),
        conditioner,
        point,
        out);
    out.weight.cdf =
        enoki::arange<decltype(out.weight.cdf)>(enoki::slices(out.weight.cdf));
    std::sort(
        out.weight.cdf.begin(),
        out.weight.cdf.end(),
        [&out](ScalarS idx1, ScalarS idx2) {
            return out.weight.pmf.coeff((size_t)idx1) <
                out.weight.pmf.coeff((size_t)idx2); // sort ascending
        });

    // Kill off weakest components
    size_t n_components = enoki::slices(out);
    for (size_t slice_i = 0; slice_i < n_components - max_components;
         ++slice_i) {
        size_t sorted_i = (size_t)out.weight.cdf.coeff(slice_i);
        if (sorted_i == preserve_idx) {
            --max_components;
            continue;
        }
        out.weight.pmf.coeff(sorted_i) = 0;
    }

    bool cdf_success = out.weight.prepare();
    // assert(cdf_success);
    return cdf_success;
}

template <typename Joint_, typename Marginal_, typename Conditional_>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::prepare_vectorized(
    const Joint_& joint) -> void {
    sdmm::create_marginal(joint, marginal);
    matrix_expr_t<Marginal> cov_aa_sqrt_inv =
        inverse_lower_tri(marginal.cov_sqrt);

    matrix_expr_t<Conditional> cov_bb;
    for (size_t r = MarginalSize; r < JointSize; ++r) {
        for (size_t c = MarginalSize; c < JointSize; ++c) {
            cov_bb(r - MarginalSize, c - MarginalSize) = joint.cov(r, c);
        }
    }

    typename tangent_space_t<Conditional>::EmbeddedExpr mean_b;
    for (size_t r = MarginalTangentSpace::Embedded::Size;
         r < JointTangentSpace::Embedded::Size;
         ++r) {
        mean_b.coeff(r - MarginalSize) = joint.tangent_space.mean.coeff(r);
    }
    tangent_space.set_mean(mean_b);
    spdlog::debug("tangent_space.mean={}", tangent_space.mean);

    typename matrix_expr_t<
        Joint>::template ReplaceSize<ConditionalSize, MarginalSize>
        cov_ba;
    typename matrix_expr_t<
        Joint>::template ReplaceSize<MarginalSize, ConditionalSize>
        cov_ab;
    for (size_t r = MarginalSize; r < JointSize; ++r) {
        for (size_t c = 0; c < MarginalSize; ++c) {
            cov_ba(r - MarginalSize, c) = joint.cov(r, c);
            cov_ab(c, r - MarginalSize) = joint.cov(r, c);
        }
    }
    auto aa_sqrt_inv_ab = cov_aa_sqrt_inv * cov_ab;
    conditional.cov =
        cov_bb - linalg::transpose(aa_sqrt_inv_ab) * aa_sqrt_inv_ab;
    mean_transform =
        cov_ba * linalg::transpose(cov_aa_sqrt_inv) * cov_aa_sqrt_inv;

    conditional.prepare_cov();
}

template <typename Joint_, typename Marginal_, typename Conditional_>
template <
    typename Conditional,
    std::enable_if_t<!Conditional::TangentSpace::HasTangentSpaceOffset, int>>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::
    create_conditional_vectorized(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void {
    ScalarExpr inv_jacobian_to, inv_jacobian_from;
    out.tangent_space.set_mean(tangent_space.from(
        mean_transform * marginal.tangent_space.to(point, inv_jacobian_to),
        inv_jacobian_from));
    marginal.posterior(point, out.weight.pmf);
    out.cov = conditional.cov;
    out.cov_sqrt = conditional.cov_sqrt;
    out.inv_cov_sqrt_det = conditional.inv_cov_sqrt_det;
}

template <typename Joint_, typename Marginal_, typename Conditional_>
template <
    typename Conditional,
    std::enable_if_t<Conditional::TangentSpace::HasTangentSpaceOffset, int>>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::
    create_conditional_vectorized(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void {
    ScalarExpr inv_jacobian_to, inv_jacobian_from;
    out.tangent_space = tangent_space;
    out.tangent_space.tangent_mean =
        mean_transform * marginal.tangent_space.to(point, inv_jacobian_to);
    marginal.posterior(point, out.weight.pmf);
    out.cov = conditional.cov;
    out.cov_sqrt = conditional.cov_sqrt;
    out.inv_cov_sqrt_det = conditional.inv_cov_sqrt_det;
}

template <typename Joint_, typename Marginal_, typename Conditional_>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::
    create_conditional_weights_vectorized(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void {
    marginal.posterior(point, out.weight.pmf);
}

template <typename Joint_, typename Marginal_, typename Conditional_>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::
    create_conditional_means_and_covs(
        const MarginalEmbeddedS& point,
        Conditional& out) -> void {
    ScalarExpr inv_jacobian_to, inv_jacobian_from;
    out.tangent_space.set_mean(tangent_space.from(
        mean_transform * marginal.tangent_space.to(point, inv_jacobian_to),
        inv_jacobian_from));
    out.cov = conditional.cov;
    out.cov_sqrt = conditional.cov_sqrt;
    out.inv_cov_sqrt_det = conditional.inv_cov_sqrt_det;
}

} // namespace sdmm

ENOKI_STRUCT_SUPPORT(
    sdmm::SDMMConditioner,
    marginal,
    conditional,
    mean_transform,
    tangent_space,
    marginal_tangents);
