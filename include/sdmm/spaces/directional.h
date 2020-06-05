#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/linalg/coordinate_system.h"

namespace sdmm {

template<typename Embedded_, typename Tangent_>
struct DirectionalTangentSpace {
    static_assert(
        std::is_same_v<enoki::scalar_t<Embedded_>, enoki::scalar_t<Tangent_>>
    );
    static_assert(Embedded_::Size == 3 && Tangent_::Size == 2);

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

    template<typename EmbeddedIn>
    auto to(const EmbeddedIn& embedded) const -> TangentExpr {
        return embedded - mean;
    }

    template<typename TangentIn>
    auto from(const TangentIn& tangent) const -> EmbeddedExpr {
        return tangent + mean;
    }

    auto set_mean(const Embedded& mean_) -> void { mean = mean_; }
    auto set_mean(Embedded&& mean_) -> void { mean = std::move(mean_); }

    Embedded mean;
    
    linalg::CoordinateSystem<Embedded> coordinate_system;

    ENOKI_STRUCT(DirectionalTangentSpace, mean, coordinate_system);
};

}

ENOKI_STRUCT_SUPPORT(sdmm::DirectionalTangentSpace, mean, coordinate_system)
