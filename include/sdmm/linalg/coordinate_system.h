#pragma once

#include <utility>

#include <enoki/array.h>

#include "sdmm/core/utils.h"

namespace sdmm::linalg {

// Adapted from mitsuba2
template<typename Vector>
struct CoordinateSystem {
    static_assert(Vector::Size == 3, "CoordinateSystem works on 3D vectors!");

    using Scalar = enoki::value_t<Vector>;
    using Rotation = Matrix<Scalar, 3>;
    using ScalarExpr = enoki::expr_t<Scalar>;
    using VectorExpr = enoki::expr_t<Vector>;

    auto prepare(const Vector& n_) -> void {
        /* Based on "Building an Orthonormal Basis, Revisited" by
           Tom Duff, James Burgess, Per Christensen,
           Christophe Hery, Andrew Kensler, Max Liani,
           and Ryusuke Villemin (JCGT Vol 6, No 1, 2017) */

        ScalarExpr sign = enoki::sign(n_.z()),
              a    = -enoki::rcp(sign + n_.z()),
              b    = n_.x() * n_.y() * a;

        from.col(0) = VectorExpr(
            enoki::mulsign(enoki::sqr(n_.x()) * a, n_.z()) + 1.f,
            enoki::mulsign(b, n_.z()),
            enoki::mulsign_neg(n_.x(), n_.z())
        );
        from.col(1) = VectorExpr(b, sign + enoki::sqr(n_.y()) * a, -n_.y());
        from.col(2) = n_;
        // from = Rotation::from_cols(s, t, n);
        to = linalg::transpose(from);
    }

    Rotation to;
    Rotation from;

    ENOKI_STRUCT(CoordinateSystem, to, from);
};

}

ENOKI_STRUCT_SUPPORT(sdmm::linalg::CoordinateSystem, to, from);
