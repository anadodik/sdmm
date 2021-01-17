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

    using ScalarExpr = enoki::expr_t<Scalar>;
    using EmbeddedExpr = enoki::expr_t<Embedded>;
    using TangentExpr = enoki::expr_t<Tangent>;
    using DirectionalEmbeddedExpr = enoki::expr_t<DirectionalEmbedded>;
    using DirectionalTangentExpr = enoki::expr_t<DirectionalTangent>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using EmbeddedS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using TangentS = sdmm::Vector<ScalarS, enoki::array_size_v<Tangent>>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    constexpr static bool IsEuclidian = false;
    constexpr static bool HasTangentSpaceOffset = false;

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
        return EmbeddedExpr(
            (tangent.coeff(Indices) + mean.coeff(Indices))...,
            directional1,
            directional2,
            directional3
        );
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
        const DirectionalEmbeddedExpr directional_local =
            coordinate_system.to * directional(embedded);
        const ScalarExpr cos_angle = directional_local.z();
        auto length = enoki::norm(directional_local);
        if(enoki::any(enoki::abs(length - 1) >= 1e-4)) {
            std::cerr << fmt::format("directional_local.length()={},\n{}\n{}\n", length, embedded, directional_local);
            assert(enoki::all(cos_angle >= -1 - 1e-4));
        }

        const ScalarExpr angle = enoki::safe_acos(cos_angle);
        const ScalarExpr sin_angle = enoki::safe_sqrt(1 - cos_angle * cos_angle);
        const ScalarExpr rcp_sinc_angle = enoki::select(
            sin_angle < 1e-4,
            ScalarExpr(1),
            angle / sin_angle
        );

        inv_jacobian = enoki::select(cos_angle <= -1, 0, rcp_sinc_angle);
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
            sin_angle / length
        );

        inv_jacobian = enoki::select(length < M_PI, sinc_angle, 0);
        const DirectionalEmbeddedExpr embedded_local{
            directional_tangent.x() * sinc_angle,
            directional_tangent.y() * sinc_angle,
            cos_angle
        };
        const DirectionalEmbeddedExpr embedded = coordinate_system.from * embedded_local;

        return from_unwrap(
            tangent,
            std::make_index_sequence<Tangent::Size - 2>{},
            embedded.x(),
            embedded.y(),
            embedded.z()
        );
    }

    auto to_center_jacobian() const -> sdmm::Matrix<ScalarExpr, Tangent::Size, Embedded::Size> {
        using DirectionalJacobian = sdmm::Matrix<ScalarExpr, 2, 3>;
        using Jacobian = sdmm::Matrix<ScalarExpr, Tangent::Size, Embedded::Size>;
        DirectionalJacobian directional_jacobian = enoki::zero<DirectionalJacobian>();
        Jacobian jacobian = enoki::zero<Jacobian>();
        for(size_t diag_i = 0; diag_i < 2; ++diag_i) {
            directional_jacobian(diag_i, diag_i) = enoki::full<ScalarExpr>(
                1.f, enoki::slices(directional_jacobian)
            );
        }
        directional_jacobian = directional_jacobian * coordinate_system.to;
        for(size_t diag_i = 0; diag_i < Tangent::Size - 2; ++diag_i) {
            jacobian(diag_i, diag_i) = enoki::full<ScalarExpr>(
                1.f, enoki::slices(jacobian)
            );
        }
        for(size_t r = 0; r < 2; ++r) {
            for(size_t c = 0; c < 3; ++c) {
                jacobian(r + Tangent::Size - 2, c + Embedded::Size - 3) = directional_jacobian(r, c);
            }
        }
        return jacobian;
    }

    // Calculates the Jacobian matrix approximation for the transformation
    // \log_{\mu} ( \vec{t} )
    template<typename TangentIn>
    auto from_jacobian(
        const TangentIn& tangent
    ) const -> std::pair<EmbeddedExpr, sdmm::Matrix<ScalarExpr, Embedded::Size, Tangent::Size>> {
        constexpr static ScalarS cos_angle_min = -0.98;

        using DirectionalJacobian = sdmm::Matrix<ScalarExpr, 3, 2>;
        using Jacobian = sdmm::Matrix<ScalarExpr, Embedded::Size, Tangent::Size>;
        DirectionalJacobian directional_jacobian = enoki::zero<DirectionalJacobian>();
        Jacobian jacobian = enoki::zero<Jacobian>();

        const DirectionalTangentExpr directional_tangent = directional(tangent);
        const ScalarExpr length_sqr = enoki::squared_norm(directional_tangent);
        const ScalarExpr length = enoki::sqrt(length_sqr);
        auto [sin_angle, cos_angle] = enoki::sincos(length);
        const ScalarExpr sinc_angle = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            ScalarExpr(1),
            sin_angle / length
        );

        ScalarExpr cos_minus_sinc_over_length_sqr = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            0,
            (cos_angle - sinc_angle) / length_sqr
        );
        directional_jacobian(0, 0) = sinc_angle + directional_tangent.coeff(0) * directional_tangent.coeff(0) * cos_minus_sinc_over_length_sqr;
        directional_jacobian(1, 1) = sinc_angle + directional_tangent.coeff(1) * directional_tangent.coeff(1) * cos_minus_sinc_over_length_sqr;

        ScalarExpr off_diagonal = directional_tangent.coeff(0) * directional_tangent.coeff(1) * cos_minus_sinc_over_length_sqr;
        directional_jacobian(0, 1) = off_diagonal;
        directional_jacobian(1, 0) = off_diagonal;

        directional_jacobian(2, 0) = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            0, -directional_tangent.coeff(0) * sinc_angle
        );
        directional_jacobian(2, 1) = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            0, -directional_tangent.coeff(1) * sinc_angle
        );
        directional_jacobian = coordinate_system.from * directional_jacobian;
        for(size_t diag_i = 0; diag_i < Tangent::Size - 2; ++diag_i) {
            jacobian(diag_i, diag_i) = enoki::full<ScalarExpr>(
                1.f, enoki::slices(jacobian)
            );
        }
        for(size_t r = 0; r < 3; ++r) {
            for(size_t c = 0; c < 2; ++c) {
                jacobian(r + Embedded::Size - 3, c + Tangent::Size - 2) = directional_jacobian(r, c);
            }
        }

        const DirectionalEmbeddedExpr embedded_local{
            directional_tangent.x() * sinc_angle,
            directional_tangent.y() * sinc_angle,
            cos_angle
        };
        const DirectionalEmbeddedExpr directional_embedded = coordinate_system.from * embedded_local;
        EmbeddedExpr embedded = from_unwrap(
            tangent,
            std::make_index_sequence<Tangent::Size - 2>{},
            directional_embedded.x(),
            directional_embedded.y(),
            directional_embedded.z()
        );

        return {embedded, jacobian};
    }

    auto set_mean(const Embedded& mean_) -> void {
        mean = mean_;
        DirectionalEmbeddedExpr directional_mean(
            mean.coeff(Embedded::Size - 3),
            mean.coeff(Embedded::Size - 2),
            mean.coeff(Embedded::Size - 1)
        );
        directional_mean = enoki::normalize(directional_mean);
        coordinate_system.prepare(directional_mean);
    }

    Embedded mean;
    // TODO: fix template to take Scalar
    linalg::CoordinateSystem<sdmm::Vector<Scalar, 3>> coordinate_system;

    ENOKI_STRUCT(SpatioDirectionalTangentSpace, mean, coordinate_system);
};

template<typename Embedded, typename Tangent>
void to_json(json& j, const SpatioDirectionalTangentSpace<Embedded, Tangent>& tangent_space) {
    j = json{
        {"tangent_space.mean", tangent_space.mean},
        {"tangent_space.coordinate_system", tangent_space.coordinate_system},
    };
}

template<typename Embedded, typename Tangent>
void from_json(const json& j, SpatioDirectionalTangentSpace<Embedded, Tangent>& tangent_space) {
    j.at("tangent_space.mean").get_to(tangent_space.mean);
    j.at("tangent_space.coordinate_system").get_to(tangent_space.coordinate_system);
}

}

ENOKI_STRUCT_SUPPORT(sdmm::SpatioDirectionalTangentSpace, mean, coordinate_system)
