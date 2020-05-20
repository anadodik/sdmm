#pragma once

#include <enoki/array.h>
#include <enoki/matrix.h>

namespace sdmm::linalg {

template<typename Value, size_t Size>
void cholesky(
    const enoki::Matrix<Value, Size>& in,
    enoki::Matrix<Value, Size>& out,
    enoki::mask_t<Value>& is_psd
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

}
