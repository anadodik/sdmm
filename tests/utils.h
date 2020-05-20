#pragma once

#include <Eigen/Dense>

#include "sdmm/core/constants.h"

template<typename T, typename U>
bool approx_equals(const T& first, const U& second) {
    return enoki::all_nested(
        enoki::abs(first - second) <=
        sdmm::epsilon<enoki::scalar_t<T>>
    );
}

template<typename T, typename Derived>
bool approx_equals_lower_tri(
    const T& enoki_result,
    const Eigen::MatrixBase<Derived>& eigen_result,
    int mat_i=0
) {
    static_assert(Derived::RowsAtCompileTime == T::Size);
    static_assert(Derived::ColsAtCompileTime == T::Size);
    static constexpr size_t Size = T::Size;

    bool all_equal = true;
    for(size_t r = 0; r < Size; ++r) {
        for(size_t c = 0; c < r + 1; ++c) {
            auto abs_diff = enoki::abs(
                eigen_result(r, c) - enoki_result(r, c).coeff(mat_i)
            );
            all_equal = all_equal && (abs_diff < sdmm::epsilon<float>);
        }
    }
    return all_equal;
}
