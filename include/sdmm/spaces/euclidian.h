#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"

namespace sdmm {

template <typename Embedded_, typename Tangent_>
struct EuclidianTangentSpace {
    static_assert(
        std::is_same_v<enoki::scalar_t<Embedded_>, enoki::scalar_t<Tangent_>>);

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

    constexpr static bool IsEuclidian = true;
    constexpr static bool HasTangentSpaceOffset = false;

    template <typename EmbeddedIn>
    auto to(const EmbeddedIn& embedded, ScalarExpr& inv_jacobian) const
        -> TangentExpr {
        inv_jacobian = ScalarExpr(1);
        return embedded - mean;
    }

    template <typename TangentIn>
    auto from(const TangentIn& tangent, ScalarExpr& inv_jacobian) const
        -> EmbeddedExpr {
        inv_jacobian = ScalarExpr(1);
        return tangent + mean;
    }

    auto set_mean(const Embedded& mean_) -> void {
        mean = mean_;
    }
    auto set_mean(Embedded&& mean_) -> void {
        mean = std::move(mean_);
    }

    Embedded mean;

    // Dummy necessary because Enoki does not allow structs with one parameter.
    Mask dummy;

    ENOKI_STRUCT(EuclidianTangentSpace, mean, dummy);
};

template <typename Embedded, typename Tangent>
void to_json(
    json& j,
    const EuclidianTangentSpace<Embedded, Tangent>& tangent_space) {
    j = json{
        {"tangent_space.mean", tangent_space.mean},
    };
}

template <typename Embedded, typename Tangent>
void from_json(
    const json& j,
    EuclidianTangentSpace<Embedded, Tangent>& tangent_space) {
    j.at("tangent_space.mean").get_to(tangent_space.mean);
}

} // namespace sdmm

ENOKI_STRUCT_SUPPORT(sdmm::EuclidianTangentSpace, mean, dummy)
