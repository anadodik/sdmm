#pragma once

#include <enoki/array.h>

namespace sdmm {

template<typename Value> struct epsilon_t {
    static constexpr Value value = epsilon_t<enoki::scalar_t<Value>>::value;
};
template<> struct epsilon_t<float> {
    static constexpr float value = 1e-5f;
};
template<> struct epsilon_t<double> {
    static constexpr double value = 1e-12f;
};

template<typename Value>
inline constexpr auto epsilon = epsilon_t<Value>::value;

template<typename Value> struct inv_sqrt_2_pi_t {
    static constexpr Value value = 
        inv_sqrt_2_pi_t<enoki::scalar_t<Value>>::value;
};

template<> struct inv_sqrt_2_pi_t<float> {
    static constexpr float value = 0.3989422804f;
};

template<> struct inv_sqrt_2_pi_t<double> {
    static constexpr double value = 
        0.39894228040143267793994605993438186847585863116492;
};

template<typename Value>
inline constexpr auto inv_sqrt_2_pi = inv_sqrt_2_pi_t<Value>::value;

// TODO: try pushing constexpr math upstream
template<typename Value, size_t CovSize>
inline const auto gaussian_normalization =
    enoki::pow(inv_sqrt_2_pi<Value>, CovSize);

}
