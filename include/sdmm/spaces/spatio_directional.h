#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/linalg/coordinate_system.h"

namespace sdmm {

template<typename Embedded_, typename Tangent_>
struct SpatioDirectionalTangentSpace {
    static_assert(
        std::is_same_v<enoki::scalar_t<Embedded_>, enoki::scalar_t<Tangent_>>
    );
    static_assert(Embedded_::Size >= 3 && Tangent_::Size == Embedded_::Size - 1);

    using Scalar = enoki::value_t<Embedded_>;
    using Embedded = Embedded_;
    using Tangent = Tangent_;
    using DirectionalEmbedded = sdmm::Vector<Scalar, 3>;
    using DirectionalTangent = sdmm::Vector<Scalar, 2>;
    using Mask = enoki::mask_t<Embedded_>;
    using Matrix = enoki::Matrix<Scalar, 3>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using EmbeddedExpr = enoki::expr_t<Embedded>;
    using TangentExpr = enoki::expr_t<Tangent>;
    using DirectionalEmbeddedExpr = enoki::expr_t<DirectionalEmbedded>;
    using DirectionalTangentExpr = enoki::expr_t<DirectionalTangent>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using EmbeddedS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using TangentS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    template<typename EmbeddedIn, size_t... Indices>
    auto to_unwrap(
        const EmbeddedIn& embedded,
        [[maybe_unused]] std::index_sequence<Indices...> index_sequence,
        const ScalarExpr& directional1,
        const ScalarExpr& directional2
    ) const -> TangentExpr {
        return {
            (embedded.coeff(Indices) - mean.coeff(Indices))...,
            directional1,
            directional2
        };
    }

    template<typename TangentIn, size_t... Indices>
    auto from_unwrap(
        const TangentIn& tangent,
        [[maybe_unused]] std::index_sequence<Indices...> index_sequence,
        const ScalarExpr& directional1,
        const ScalarExpr& directional2,
        const ScalarExpr& directional3
    ) const -> EmbeddedExpr {
        return {
            (tangent.coeff(Indices) + mean.coeff(Indices))...,
            directional1,
            directional2,
            directional3
        };
    }

    template<typename EmbeddedIn, std::enable_if_t<EmbeddedIn::Size == Embedded::Size, int> = 0>
    auto directional(const EmbeddedIn& embedded) const -> DirectionalEmbeddedExpr {
        return {
            embedded.coeff(Embedded::Size - 3),
            embedded.coeff(Embedded::Size - 2),
            embedded.coeff(Embedded::Size - 1)
        };
    }

    template<typename TangentIn, std::enable_if_t<TangentIn::Size == Tangent::Size, int> = 0>
    auto directional(const TangentIn& tangent) const -> DirectionalTangentExpr {
        return {
            tangent.coeff(Tangent::Size - 2),
            tangent.coeff(Tangent::Size - 1)
        };
    }

    template<typename EmbeddedIn>
    auto to(const EmbeddedIn& embedded, ScalarExpr& inv_jacobian) const -> TangentExpr {
        DirectionalEmbeddedExpr directional_local =
            coordinate_system.from * directional(embedded);
        const ScalarExpr cos_angle = directional_local.z();
        assert(enoki::all(cos_angle >= -1));

        const ScalarExpr angle = enoki::safe_acos(cos_angle);
        const ScalarExpr sin_angle = enoki::safe_sqrt(1 - cos_angle * cos_angle);
        const ScalarExpr rcp_sinc_angle = enoki::select(
            sin_angle < 1e-4,
            ScalarExpr(1),
            angle / sin_angle
        );

        inv_jacobian = rcp_sinc_angle;
        return to_unwrap(
            embedded,
            std::make_index_sequence<Embedded::Size - 3>{},
            directional_local.x() * rcp_sinc_angle,
            directional_local.y() * rcp_sinc_angle
        );
    }

    template<typename TangentIn>
    auto from(const TangentIn& tangent, ScalarExpr& inv_jacobian) const -> EmbeddedExpr {
        DirectionalTangentExpr directional_tangent = directional(tangent);
        ScalarExpr length = enoki::norm(directional_tangent);
        // if(length >= M_PI) {
        //     embedding = EmbeddingVectord::Zero();
        //     return false;
        // }
        auto [sin_angle, cos_angle] = enoki::sincos(length);
        const ScalarExpr sinc_angle = enoki::select(
            sin_angle < 1e-4,
            ScalarExpr(1),
            sinc_angle / length
        );

        inv_jacobian = enoki::select(length > M_PI, sinc_angle, 0);
        const DirectionalEmbeddedExpr embedded_local{
            directional_tangent.x() * sinc_angle,
            directional_tangent.y() * sinc_angle,
            cos_angle
        };
        const DirectionalEmbeddedExpr embedded = coordinate_system.from * embedded_local;

        return from_unwrap(
            tangent,
            std::make_index_sequence<Embedded::Size - 3>{},
            embedded.x(),
            embedded.y(),
            embedded.z()
        );
    }

    auto set_mean(const Embedded& mean_) -> void {
        mean = mean_;
        coordinate_system.prepare({
            mean.coeff(Embedded::Size - 3),
            mean.coeff(Embedded::Size - 2),
            mean.coeff(Embedded::Size - 1),
        });
    }

    auto set_mean(Embedded&& mean_) -> void {
        mean = std::move(mean_);
        coordinate_system.prepare({
            mean.coeff(Embedded::Size - 3),
            mean.coeff(Embedded::Size - 2),
            mean.coeff(Embedded::Size - 1),
        });
        // coordinate_system.prepare(mean);
    }

    Embedded mean;
    // TODO: fix template to take Scalar
    linalg::CoordinateSystem<sdmm::Vector<Scalar, 3>> coordinate_system;

    ENOKI_STRUCT(SpatioDirectionalTangentSpace, mean, coordinate_system);
};

}

ENOKI_STRUCT_SUPPORT(sdmm::SpatioDirectionalTangentSpace, mean, coordinate_system)
