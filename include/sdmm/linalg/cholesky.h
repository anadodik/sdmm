#pragma once

#include <enoki/array.h>
#include <enoki/matrix.h>

#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/utils.h"

namespace sdmm::linalg {

template<typename Value_, size_t Size, typename Value=std::decay_t<Value_>>
void cholesky(
    const enoki::Matrix<Value_, Size>& in,
    enoki::Matrix<Value_, Size>& out,
    enoki::mask_t<Value_>& is_psd
) {
    using Matrix = enoki::Matrix<Value, Size>;
    using Mask = enoki::mask_t<Value>;

    out = enoki::zero<Matrix>();
    is_psd = Mask(true);
    for(size_t r = 0; r < Size; ++r) {
        for(size_t c = 0; c < r; ++c) {
            Value ksum = enoki::zero<Value>();
            for(size_t k = 0; k < c; ++k) {
                ksum += out(r, k) * out(c, k);
            }
            out(r, c) = (in(r, c) - ksum) / out(c, c);
        }
        Matrix out_t = enoki::transpose(out);
        Value row_sum = enoki::hsum(out_t.col(r) * out_t.col(r));
        Value value = in(r, r) - row_sum;
        is_psd = is_psd && (value > 0);
        out(r, r) = enoki::sqrt(value);
    }
}

template<
    typename ValueLHS_,
    typename ValueRHS_,
    size_t Size,
    typename ValueLHS = enoki::expr_t<ValueLHS_>,
    typename ValueRHS = enoki::expr_t<ValueRHS_>
>
Vector<ValueLHS, Size> solve(
    const Matrix<ValueLHS_, Size>& L,
    const Vector<ValueRHS_, Size>& b
) {
    Vector<ValueLHS, Size> x;
    x[0] = b[0] / L(0, 0);
    for(size_t r = 1; r < Size; ++r) {
        ValueLHS numerator = b[r];
        for(size_t c = 0; c < r; ++c) {
            numerator -= L(r, c) * x[c];
        }
        x[r] = numerator / L(r, r);
        assert(enoki::any(L(r, r) != ValueLHS(0)));
    }
    return x;
}

}
