#pragma once

#include <enoki/array.h>

namespace sdmm::linalg {

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

}
