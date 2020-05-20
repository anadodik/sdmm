#pragma once

#include <enoki/array.h>

namespace sdmm::linalg {

template<typename Value> constexpr auto epsilon() { return epsilon<enoki::scalar_t<Value>>(); }
template<> constexpr auto epsilon<float>() { return 1e-4f; }
template<> constexpr auto epsilon<double>() { return 1e-12; }

}
