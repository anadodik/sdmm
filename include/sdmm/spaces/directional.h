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
    auto to(const EmbeddedIn& embedded, ScalarExpr& inv_jacobian) const -> TangentExpr {
        const EmbeddedExpr embedded_local = coordinate_system.to * embedded;
        const ScalarExpr cos_angle = embedded_local.z();
        // assert(enoki::all(cos_angle >= -1));

        const ScalarExpr angle = enoki::safe_acos(cos_angle);
        const ScalarExpr sin_angle = enoki::safe_sqrt(1 - cos_angle * cos_angle);
        const ScalarExpr rcp_sinc_angle = enoki::select(
            sin_angle < 1e-4,
            ScalarExpr(1),
            angle / sin_angle
        );

        inv_jacobian = rcp_sinc_angle;
        return TangentExpr{
            embedded_local.x() * rcp_sinc_angle,
            embedded_local.y() * rcp_sinc_angle
        };
    }

    template<typename TangentIn>
    auto from(const TangentIn& tangent, ScalarExpr& inv_jacobian) const -> EmbeddedExpr {
        // if(length >= M_PI) {
        //     embedding = EmbeddingVectord::Zero();
        //     return false;
        // }
        ScalarExpr length = enoki::norm(tangent);
        auto [sin_angle, cos_angle] = enoki::sincos(length);
        const ScalarExpr sinc_angle = enoki::select(
            sin_angle < 1e-4,
            ScalarExpr(1),
            sin_angle / length
        );
        inv_jacobian = enoki::select(length < M_PI, sinc_angle, 0);

        const EmbeddedExpr embedded_local{
            tangent.x() * sinc_angle,
            tangent.y() * sinc_angle,
            cos_angle
        };

        return coordinate_system.from * embedded_local;
    }

    auto set_mean(const Embedded& mean_) -> void {
        mean = mean_;
        coordinate_system.prepare(mean);
    }

    auto set_mean(Embedded&& mean_) -> void {
        mean = std::move(mean_);
        coordinate_system.prepare(mean);
    }

    Embedded mean;
    linalg::CoordinateSystem<Embedded> coordinate_system;

    ENOKI_STRUCT(DirectionalTangentSpace, mean, coordinate_system);
};

}

ENOKI_STRUCT_SUPPORT(sdmm::DirectionalTangentSpace, mean, coordinate_system)
