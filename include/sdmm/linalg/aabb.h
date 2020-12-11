#pragma once

#include <sdmm/core/utils.h>
#include <sdmm/linalg/vector.h>

namespace sdmm::linalg {

template<typename Scalar, int Size>
struct AABB {
    using Point = sdmm::Vector<Scalar, Size>;

    AABB() {}
    AABB(Point min_, Point max_) : min(min_), max(max_) { }
    AABB(const AABB& other) = default;
    AABB(AABB&& other) = default;
    AABB& operator=(const AABB& other) = default;
    AABB& operator=(AABB&& other) = default;

    template<typename PointIn>
    auto contains(const PointIn& point) const -> bool {
        bool result = enoki::all(point > min) && enoki::all(point < max);
        return result;
    }

    auto diagonal() const -> Point {
        return max - min;
    }

    Point min;
    Point max;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(AABB, min, max);
};

}
