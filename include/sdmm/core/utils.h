#pragma once

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <enoki/array.h>
#include <enoki/dynamic.h>

#include "sdmm/linalg/matrix.h"

#define VECTORIZE_WRAP(FUNC_NAME) [](auto&&... params) { FUNC_NAME(params...); }
#define VECTORIZE_WRAP_OUTPUT(FUNC_NAME) [](auto&& output, auto&&... params) { output = FUNC_NAME(params...); }

#define VECTORIZE_WRAP_MEMBER(FUNC_NAME) \
    [](auto&& obj, [[maybe_unused]] auto&&... params) {\
        return std::forward<decltype(obj)>(obj).FUNC_NAME(params...); \
    }

namespace sdmm {

template<typename Func, typename... Args>
auto vectorize(const Func& func, Args&&... args) {
    std::tuple packet_sizes{enoki::packets(args)...};
    std::tuple slice_sizes{enoki::slices(args)...};
    spdlog::info("{}", packet_sizes);
}

template<typename Value, typename Enable=void>
struct nested_packet_size;

template<typename Value>
struct nested_packet_size<Value, std::enable_if_t<enoki::is_dynamic_v<Value>>> {
    static constexpr size_t value = Value::PacketSize;
};

template<typename Value>
struct nested_packet_size<Value, std::enable_if_t<!enoki::is_dynamic_v<Value>>> {
    static constexpr size_t value = Value::Size;
};

template<typename Value>
static constexpr size_t nested_packet_size_v = nested_packet_size<Value>::value;

template<typename Value, typename Enable=void>
struct nested_packet;

template<typename Value>
struct nested_packet<Value, std::enable_if_t<enoki::is_dynamic_v<Value>>> {
    using type = typename Value::Packet;
};

template<typename Value>
struct nested_packet<Value, std::enable_if_t<!enoki::is_dynamic_v<Value>>> {
    using type = Value;
};

template<typename Value>
using nested_packet_t = typename nested_packet<Value>::type;

template<typename Value, typename New, typename Enable=void>
struct outer_type {
    using type = New;
};

template<typename Value, typename New>
struct outer_type<
    Value,
    New,
    std::enable_if_t<
        enoki::is_array_v<Value> && (enoki::array_depth_v<Value> > 1)
    >
> {
    using type = std::remove_reference_t<typename Value::template ReplaceValue<New>>;
};

template<typename Value, typename New>
struct outer_type<
    Value,
    New,
    std::enable_if_t<
        !enoki::is_array_v<Value> ||
        (enoki::is_array_v<Value> && enoki::array_depth_v<Value> <= 1)
    >
> {
    using type = New;
};

template<typename Value, typename New>
using outer_type_t = typename outer_type<std::remove_reference_t<Value>, New>::type;

template<typename Value, std::enable_if_t<!enoki::is_array_v<Value>, int> = 0> 
auto& coeff_safe(Value& value, [[maybe_unused]] size_t i) {
    assert(i == 0);
    return value;
};

template<typename Value, std::enable_if_t<enoki::is_array_v<Value>, int> = 0> 
auto& coeff_safe(Value& value, size_t i) {
    return value.coeff(i);
};

// Taken from Mitsuba2:
template <typename Value_, size_t Size_>
struct Vector : enoki::StaticArrayImpl<Value_, Size_, false, Vector<Value_, Size_>> {
    using Base = enoki::StaticArrayImpl<Value_, Size_, false, Vector<Value_, Size_>>;

    /// Helper alias used to implement type promotion rules
    template <typename T> using ReplaceValue = Vector<T, Size_>;

    using ArrayType = Vector;
    using MaskType = enoki::Mask<Value_, Size_>;

    static constexpr bool IsMatrix = false;

    ENOKI_ARRAY_IMPORT(Base, Vector)
};

template<typename Value, size_t Rows, size_t Cols=Rows>
using Matrix = sdmm::linalg::Matrix<Value, Rows, Cols>;

// https://stackoverflow.com/questions/81870/is-it-possible-to-print-a-variables-type-in-standard-c
template <class T>
constexpr std::string_view type_name() {
    using namespace std;
#ifdef __clang__
    string_view p = __PRETTY_FUNCTION__;
    return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
    string_view p = __PRETTY_FUNCTION__;
#  if __cplusplus < 201402
    return string_view(p.data() + 36, p.size() - 36 - 1);
#  else
    return string_view(p.data() + 49, p.find(';', 49) - 49);
#  endif
#elif defined(_MSC_VER)
    string_view p = __FUNCSIG__;
    return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}

}
