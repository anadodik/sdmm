#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"

namespace sdmm {

template<typename Embedded_, typename Tangent_>
struct EuclidianTangentSpace {
    static_assert(
        std::is_same_v<enoki::scalar_t<Embedded_>, enoki::scalar_t<Tangent_>>
    );

    using Scalar = enoki::value_t<Embedded_>;
    using Embedded = Embedded_;
    using Tangent = Tangent_;
    using Mask = enoki::mask_t<Embedded_>;
    using Matrix = enoki::Matrix<Scalar, 3>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using EmbeddedExpr = enoki::expr_t<Embedded>;
    using TangentExpr = enoki::expr_t<Tangent>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using EmbeddedS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using TangentS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    TangentExpr to(const EmbeddedS& embedded) const { return embedded - mean; }

    EmbeddedExpr from(const TangentS& tangent) const { return tangent + mean; }

    void set_mean(const Embedded& mean_) { mean = mean_; }
    void set_mean(Embedded&& mean_) { mean = std::move(mean_); }

    Embedded mean;

    // Dummy necessary because Enoki does not allow structs with one parameter.
    Mask dummy;

    ENOKI_STRUCT(EuclidianTangentSpace, mean, dummy);
};

}

ENOKI_STRUCT_SUPPORT(sdmm::EuclidianTangentSpace, mean, dummy)