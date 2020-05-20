#pragma once

#include "sdmm/core/constants.h"

template<typename T, typename U>
bool enoki_approx_equals(const T& first, const U& second) {
    return enoki::all_nested(
        enoki::abs(first - second) <=
        sdmm::epsilon<enoki::scalar_t<T>>
    );
}
