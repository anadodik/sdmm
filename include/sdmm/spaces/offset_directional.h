#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/linalg/coordinate_system.h"

namespace sdmm {

template <typename Embedded_, typename Tangent_>
struct OffsetDirectionalTangentSpace {
    static_assert(
        std::is_same_v<enoki::scalar_t<Embedded_>, enoki::scalar_t<Tangent_>>);
    static_assert(Embedded_::Size == 3 && Tangent_::Size == 2);

    using Scalar = enoki::value_t<Embedded_>;
    using Embedded = Embedded_;
    using Tangent = Tangent_;
    using Mask = enoki::mask_t<Embedded_>;
    using Matrix = sdmm::Matrix<Scalar, 3>;

    using ScalarExpr = enoki::expr_t<Scalar>;
    using EmbeddedExpr = enoki::expr_t<Embedded>;
    using TangentExpr = enoki::expr_t<Tangent>;
    using MatrixExpr = enoki::expr_t<Matrix>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using EmbeddedS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using TangentS = sdmm::Vector<ScalarS, enoki::array_size_v<Tangent>>;
    using MatrixS = sdmm::Matrix<ScalarS, 3>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    constexpr static bool IsEuclidian = false;
    constexpr static bool HasTangentSpaceOffset = true;

    template <typename EmbeddedIn>
    auto to(const EmbeddedIn& embedded, ScalarExpr& inv_jacobian) const
        -> TangentExpr {
        constexpr static ScalarS cos_angle_min = -0.98;

        const EmbeddedExpr embedded_local = coordinate_system.to * embedded;
        const ScalarExpr cos_angle = embedded_local.z();
        // assert(enoki::all(cos_angle >= -1));

        const ScalarExpr angle = enoki::safe_acos(cos_angle);
        const ScalarExpr sin_angle =
            enoki::safe_sqrt(1 - cos_angle * cos_angle);
        const ScalarExpr rcp_sinc_angle =
            enoki::select(sin_angle < 1e-4, ScalarExpr(1), angle / sin_angle);

        inv_jacobian =
            enoki::select(cos_angle < cos_angle_min, 0, rcp_sinc_angle);

        return TangentExpr{
            embedded_local.x() * rcp_sinc_angle - tangent_mean.x(),
            embedded_local.y() * rcp_sinc_angle - tangent_mean.y()};
    }

    template <typename TangentIn>
    auto from(const TangentIn& tangent, ScalarExpr& inv_jacobian) const
        -> EmbeddedExpr {
        constexpr static ScalarS max_length = M_PI - 1e-1;

        TangentExpr tangent_offset = tangent + tangent_mean;
        ScalarExpr length = enoki::norm(tangent_offset);
        auto [sin_angle, cos_angle] = enoki::sincos(length);
        const ScalarExpr sinc_angle = enoki::select(
            (sin_angle < 1e-2) || length > max_length,
            ScalarExpr(1),
            sin_angle / length);
        inv_jacobian = enoki::select(length > max_length, 0, sinc_angle);

        const EmbeddedExpr embedded_local{
            tangent_offset.x() * sinc_angle,
            tangent_offset.y() * sinc_angle,
            cos_angle};

        return coordinate_system.from * embedded_local;
    }

    auto set_mean(const Embedded& mean_) -> void {
        mean = enoki::normalize(mean_);
        coordinate_system.prepare(mean);
    }

    auto rotate_to_wo(const EmbeddedS& wi) -> void {
        ScalarS neg_phi = -std::atan2(wi.y(), wi.x());
        MatrixS rotation(
            std::cos(neg_phi),
            -std::sin(neg_phi),
            0,
            std::sin(neg_phi),
            std::cos(neg_phi),
            0,
            0,
            0,
            1);
        MatrixS inv_rotation = linalg::transpose(rotation);
        coordinate_system.to = coordinate_system.to * rotation;
        coordinate_system.from = inv_rotation * coordinate_system.from;
        mean = inv_rotation * mean;
    }

    auto to_center_jacobian() const -> sdmm::Matrix<ScalarExpr, 2, 3> {
        using Jacobian = sdmm::Matrix<ScalarExpr, 2, 3>;
        Jacobian jacobian = enoki::zero<Jacobian>();
        jacobian(0, 0) = enoki::full<ScalarExpr>(1.f, enoki::slices(jacobian));
        jacobian(1, 1) = enoki::full<ScalarExpr>(1.f, enoki::slices(jacobian));
        return jacobian * coordinate_system.to;
    }

    auto from_center_jacobian() const -> sdmm::Matrix<ScalarExpr, 3, 2> {
        using Jacobian = sdmm::Matrix<ScalarExpr, 3, 2>;
        Jacobian jacobian = enoki::zero<Jacobian>();
        jacobian(0, 0) = enoki::full<ScalarExpr>(1.f, enoki::slices(jacobian));
        jacobian(1, 1) = enoki::full<ScalarExpr>(1.f, enoki::slices(jacobian));
        return coordinate_system.from * jacobian;
    }

    template <typename EmbeddedIn>
    auto to_jacobian(const EmbeddedIn& embedded) const
        -> std::pair<TangentExpr, sdmm::Matrix<ScalarExpr, 2, 3>> {
        ScalarExpr inv_jacobian;
        return to_jacobian(embedded, inv_jacobian);
    }

    // Calculates the Jacobian matrix approximation for the transformation
    // \exp_{\mu} ( \vec{x} )
    template <typename EmbeddedIn>
    auto to_jacobian(const EmbeddedIn& embedded, ScalarExpr& inv_jacobian) const
        -> std::pair<TangentExpr, sdmm::Matrix<ScalarExpr, 2, 3>> {
        using Jacobian = sdmm::Matrix<ScalarExpr, 2, 3>;
        Jacobian jacobian;

        const EmbeddedExpr embedded_local = coordinate_system.to * embedded;
        const ScalarExpr cos_angle = embedded_local.z();

        const ScalarExpr angle = enoki::safe_acos(cos_angle);
        const ScalarExpr sin_angle_sqr = 1 - cos_angle * cos_angle;
        const ScalarExpr sin_angle = enoki::safe_sqrt(sin_angle_sqr);
        const ScalarExpr rcp_sinc_angle = enoki::select(
            sin_angle < 1e-2 || cos_angle < -0.99,
            ScalarExpr(1),
            angle / sin_angle);
        inv_jacobian = enoki::select(cos_angle <= -0.99, 0, rcp_sinc_angle);

        jacobian(0, 0) = rcp_sinc_angle;
        jacobian(1, 1) = rcp_sinc_angle;

        jacobian(0, 1) = enoki::zero<ScalarExpr>(enoki::slices(rcp_sinc_angle));
        jacobian(1, 0) = enoki::zero<ScalarExpr>(enoki::slices(rcp_sinc_angle));

        ScalarExpr inv_sin_angle_sqr = enoki::select(
            sin_angle < 1e-2 || cos_angle < -0.99, 0, 1 / sin_angle_sqr);
        ScalarExpr temp_expr =
            (cos_angle * rcp_sinc_angle - 1) * inv_sin_angle_sqr;
        jacobian(0, 2) = embedded_local.coeff(0) * temp_expr;
        jacobian(1, 2) = embedded_local.coeff(1) * temp_expr;

        jacobian = jacobian * coordinate_system.to;

        TangentExpr tangent{
            embedded_local.x() * rcp_sinc_angle - tangent_mean.x(),
            embedded_local.y() * rcp_sinc_angle - tangent_mean.y()};

        return {tangent, jacobian};
    }

    // Calculates the Jacobian matrix approximation for the transformation
    // \log_{\mu} ( \vec{t} )
    template <typename TangentIn>
    auto from_jacobian(const TangentIn& tangent) const
        -> std::pair<EmbeddedExpr, sdmm::Matrix<ScalarExpr, 3, 2>> {
        constexpr static ScalarS cos_angle_min = -0.98;

        using Jacobian = sdmm::Matrix<ScalarExpr, 3, 2>;
        Jacobian jacobian = enoki::zero<Jacobian>();

        TangentExpr tangent_offset = tangent + tangent_mean;
        const ScalarExpr length_sqr = enoki::squared_norm(tangent_offset);
        const ScalarExpr length = enoki::sqrt(length_sqr);
        auto [sin_angle, cos_angle] = enoki::sincos(length);
        const ScalarExpr sinc_angle = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            ScalarExpr(1),
            sin_angle / length);

        ScalarExpr cos_minus_sinc_over_length_sqr = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            0,
            (cos_angle - sinc_angle) / length_sqr);
        jacobian(0, 0) = sinc_angle +
            tangent_offset.coeff(0) * tangent_offset.coeff(0) *
                cos_minus_sinc_over_length_sqr;
        jacobian(1, 1) = sinc_angle +
            tangent_offset.coeff(1) * tangent_offset.coeff(1) *
                cos_minus_sinc_over_length_sqr;

        ScalarExpr off_diagonal = tangent_offset.coeff(0) *
            tangent_offset.coeff(1) * cos_minus_sinc_over_length_sqr;
        jacobian(0, 1) = off_diagonal;
        jacobian(1, 0) = off_diagonal;

        jacobian(2, 0) = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            0,
            -tangent_offset.coeff(0) * sinc_angle);
        jacobian(2, 1) = enoki::select(
            sin_angle < 1e-2 || cos_angle < cos_angle_min,
            0,
            -tangent_offset.coeff(1) * sinc_angle);

        jacobian = coordinate_system.from * jacobian;

        const EmbeddedExpr embedded_local{
            tangent_offset.x() * sinc_angle,
            tangent_offset.y() * sinc_angle,
            cos_angle};

        return {coordinate_system.from * embedded_local, jacobian};
    }

    Embedded mean;
    Tangent tangent_mean;
    linalg::CoordinateSystem<Embedded> coordinate_system;

    ENOKI_STRUCT(
        OffsetDirectionalTangentSpace,
        mean,
        tangent_mean,
        coordinate_system);
};

template <typename Embedded, typename Tangent>
void to_json(
    json& j,
    const OffsetDirectionalTangentSpace<Embedded, Tangent>& tangent_space) {
    j = json{
        {"tangent_space.mean", tangent_space.mean},
        {"tangent_space.tangent_mean", tangent_space.tangent_mean},
        {"tangent_space.coordinate_system", tangent_space.coordinate_system},
    };
}

template <typename Embedded, typename Tangent>
void from_json(
    const json& j,
    OffsetDirectionalTangentSpace<Embedded, Tangent>& tangent_space) {
    j.at("tangent_space.mean").get_to(tangent_space.mean);
    j.at("tangent_space.tangent_mean").get_to(tangent_space.tangent_mean);
    j.at("tangent_space.coordinate_system")
        .get_to(tangent_space.coordinate_system);
}

} // namespace sdmm

ENOKI_STRUCT_SUPPORT(
    sdmm::OffsetDirectionalTangentSpace,
    mean,
    tangent_mean,
    coordinate_system)
