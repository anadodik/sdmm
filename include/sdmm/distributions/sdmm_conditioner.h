#pragma once

#include "sdmm/distributions/sdmm.h"

namespace sdmm {

template<typename Joint, typename Marginal>
auto create_marginal(const Joint& joint, Marginal& marginal) -> void {
    static constexpr size_t MarginalSize = Marginal::Size;
    marginal.weight = joint.weight;
    for(int r = 0; r < MarginalSize; ++r) {
        marginal.tangent_space.mean.coeff(r) = joint.tangent_space.mean.coeff(r);
        for(int c = 0; c < MarginalSize; ++c) {
            marginal.cov(r, c) = joint.cov(r, c);
        }
    }
    marginal.prepare();
}

template<typename Joint_, typename Marginal_, typename Conditional_>
struct SDMMConditioner {
    using Joint = Joint_;
    using Marginal = Marginal_;
    using Conditional = Conditional_;

    static constexpr size_t JointSize = Joint::Size;
    static constexpr size_t MarginalSize = Marginal::Size;
    static constexpr size_t ConditionalSize = Conditional::Size;

    auto prepare(const Joint_& joint) -> void;

    Marginal marginal;
    Conditional conditional;
};

template<typename Joint_, typename Marginal_, typename Conditional_>
auto SDMMConditioner<Joint_, Marginal_, Conditional_>::prepare(
    const Joint_& joint
) -> void {
    create_marginal(joint, marginal);

    matrix_expr_t<Conditional> cov_bb;
    for(int r = MarginalSize; r < JointSize; ++r) {
        for(int c = MarginalSize; c < JointSize; ++c) {
            cov_bb(r - MarginalSize, c - MarginalSize) = joint.cov(r, c);
        }
    }
    // conditional.cov = cov_bb - 
}

}
