#pragma once

#include "sdmm/distributions/sdmm.h"

namespace sdmm {

template<typename Joint, typename Marginal>
inline auto create_marginal(const Joint& joint, Marginal& marginal) -> void {
    static constexpr size_t MarginalSize = Marginal::CovSize;
    using MarginalTangentSpace = tangent_space_t<Marginal>;
    marginal.weight.pmf = joint.weight.pmf;
    marginal.weight.cdf = joint.weight.cdf;

    typename MarginalTangentSpace::EmbeddedExpr mean;
    for(size_t r = 0; r < MarginalSize; ++r) {
        mean.coeff(r) = joint.tangent_space.mean.coeff(r);
        for(size_t c = 0; c < MarginalSize; ++c) {
            marginal.cov(r, c) = joint.cov(r, c);
        }
    }
    for(size_t r = 0; r < MarginalTangentSpace::Embedded::Size; ++r) {
        mean.coeff(r) = joint.tangent_space.mean.coeff(r);
    }
    marginal.tangent_space.set_mean(mean);
    marginal.prepare_cov();
}

// TODO: rename to SDMMDecomposition?
template<typename Joint_, typename Marginal_, typename Conditional_>
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

    using ScalarExpr = enoki::expr_t<enoki::value_t<vector_t<Joint>>>;
    using JointMatrix = matrix_t<Joint>;
    using MarginalVectorS = vector_s_t<Marginal>;
    using MeanTransformMatrix = typename JointMatrix::template ReplaceSize<
        ConditionalSize, MarginalSize
    >;
    using MarginalVector = typename Marginal::Vector;

    auto prepare_vectorized(const Joint& joint) -> void;
    auto create_conditional_vectorized(const MarginalVectorS& point) -> void;

    Marginal marginal;
    Conditional conditional;

    MeanTransformMatrix mean_transform;
    ConditionalTangentSpace tangent_space;
    MarginalVector marginal_tangents;

    ENOKI_STRUCT(SDMMConditioner, marginal, conditional, mean_transform, tangent_space, marginal_tangents);
};

template<typename Conditioner>
inline auto prepare(
    // cannot declare joint as const because enoki complains
    Conditioner& conditioner, typename Conditioner::Joint& joint
) -> void {
    enoki::vectorize_safe(
        VECTORIZE_WRAP_MEMBER(prepare_vectorized), std::forward<Conditioner>(conditioner), joint
    );
    assert(conditioner.marginal.weight.prepare());
}

template<typename Conditioner>
inline auto create_conditional(
    // cannot declare point as const because enoki complains
    Conditioner& conditioner, typename Conditioner::MarginalVectorS& point
) -> void {
    enoki::vectorize_safe(
        VECTORIZE_WRAP_MEMBER(create_conditional_vectorized), conditioner, point
    );
    assert(conditioner.conditional.weight.prepare());
}

template<typename Joint_, typename Marginal_, typename Conditional_>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::prepare_vectorized(
    const Joint_& joint
) -> void {
    create_marginal(joint, marginal);
    matrix_expr_t<Marginal> cov_aa_sqrt_inv = inverse_lower_tri(marginal.cov_sqrt);

    matrix_expr_t<Conditional> cov_bb;
    for(size_t r = MarginalSize; r < JointSize; ++r) {
        for(size_t c = MarginalSize; c < JointSize; ++c) {
            cov_bb(r - MarginalSize, c - MarginalSize) = joint.cov(r, c);
        }
    }

    typename tangent_space_t<Conditional>::EmbeddedExpr mean_b;
    for(
        size_t r = MarginalTangentSpace::Embedded::Size;
        r < JointTangentSpace::Embedded::Size;
        ++r
    ) {
        mean_b.coeff(r - MarginalSize) = joint.tangent_space.mean.coeff(r);
    }
    tangent_space.set_mean(mean_b);
    spdlog::debug("tangent_space.mean={}", tangent_space.mean);

    typename matrix_expr_t<Joint>::template ReplaceSize<ConditionalSize, MarginalSize> cov_ba;
    typename matrix_expr_t<Joint>::template ReplaceSize<MarginalSize, ConditionalSize> cov_ab;
    for(size_t r = MarginalSize; r < JointSize; ++r) {
        for(size_t c = 0; c < MarginalSize; ++c) {
            cov_ba(r - MarginalSize, c) = joint.cov(r, c);
            cov_ab(c, r - MarginalSize) = joint.cov(r, c);
        }
    }
    auto aa_sqrt_inv_ab = cov_aa_sqrt_inv * cov_ab;
    conditional.cov = cov_bb - linalg::transpose(aa_sqrt_inv_ab) * aa_sqrt_inv_ab;
    mean_transform = cov_ba * enoki::transpose(cov_aa_sqrt_inv) * cov_aa_sqrt_inv;

    conditional.prepare_cov();
}

template<typename Joint_, typename Marginal_, typename Conditional_>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::create_conditional_vectorized(
    const MarginalVectorS& point
) -> void {
    ScalarExpr inv_jacobian_to, inv_jacobian_from;
    conditional.tangent_space.set_mean(
        tangent_space.from(
            mean_transform * marginal.tangent_space.to(point, inv_jacobian_to),
            inv_jacobian_from
        )
    );
    marginal.posterior(point, conditional.weight.pmf, marginal_tangents);
}

}

ENOKI_STRUCT_SUPPORT(sdmm::SDMMConditioner, marginal, conditional, mean_transform, tangent_space, marginal_tangents);
