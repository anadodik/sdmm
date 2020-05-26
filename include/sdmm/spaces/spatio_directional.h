#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"

namespace sdmm {

template<typename Embedded_, typename Tangent_>
struct SDTangentSpace {
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
    using MatrixExpr = enoki::expr_t<Matrix>;
    using MaskExpr = enoki::expr_t<Mask>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using EmbeddedS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using TangentS = sdmm::Vector<ScalarS, enoki::array_size_v<Embedded>>;
    using MaskS = enoki::mask_t<ScalarS>;

    using Packet = nested_packet_t<Scalar>;

    Tangent to(const EmbeddedS& embedded) const { return embedded - mean; }

    Embedded from(const TangentS& tangent) const { return tangent + mean; }

    Embedded mean;
    Matrix to_matrix;
    Matrix from_matrix;

    ENOKI_STRUCT(SDTangentSpace, mean, to_matrix, from_matrix);
};

}

ENOKI_STRUCT_SUPPORT(sdmm::SDTangentSpace, mean, to_matrix, from_matrix);
