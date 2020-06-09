#pragma once

#include <utility>

#include <enoki/array.h>

#include "sdmm/core/utils.h"

namespace sdmm::linalg {

// Adapted from mitsuba2
template <typename Vector>
struct CoordinateSystem {
    static_assert(Vector::Size == 3, "CoordinateSystem works on 3D vectors!");

    using Float = enoki::value_t<Vector>;
    using Rotation = Matrix<Float, 3>;
    using VectorExpr = enoki::expr_t<Vector>;

    auto prepare(const Vector& n_) -> void {
        n = n_;
        /* Based on "Building an Orthonormal Basis, Revisited" by
           Tom Duff, James Burgess, Per Christensen,
           Christophe Hery, Andrew Kensler, Max Liani,
           and Ryusuke Villemin (JCGT Vol 6, No 1, 2017) */

        Float sign = enoki::sign(n.z()),
              a    = -enoki::rcp(sign + n.z()),
              b    = n.x() * n.y() * a;

        // TODO: get rid of temporaries: s, t
        s = VectorExpr(
            enoki::mulsign(enoki::sqr(n.x()) * a, n.z()) + 1.f,
            enoki::mulsign(b, n.z()),
            enoki::mulsign_neg(n.x(), n.z())
        );
        t = VectorExpr(b, sign + enoki::sqr(n.y()) * a, -n.y());
        from = Rotation::from_cols(s, t, n);
        to = linalg::transpose(from);
    }

    Vector n;
    Vector s;
    Vector t;
    Rotation to;
    Rotation from;

    ENOKI_STRUCT(CoordinateSystem, n, s, t, to, from);
};

}

ENOKI_STRUCT_SUPPORT(sdmm::linalg::CoordinateSystem, n, s, t, to, from);
