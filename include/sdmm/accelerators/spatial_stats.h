#pragma once

#include <cstdint>

#include <enoki/array.h>

namespace sdmm {

struct SpatialStats {
    using Point = sdmm::Vector<float, 3>;

    SpatialStats() {
        clear();
    }
    SpatialStats(SpatialStats& other) = delete;
    SpatialStats(const SpatialStats& other) = default;
    SpatialStats& operator=(SpatialStats& other) = delete;
    SpatialStats(SpatialStats&& other) = default;
    SpatialStats& operator=(SpatialStats&& other) = default;

    Point mean_point_;
    Point mean_sqr_point_;

    auto mean_point() const -> const Point& {
        return mean_point_;
    }
    auto mean_sqr_point() const -> const Point& {
        return mean_sqr_point_;
    }
    uint32_t size = 0;

    auto push_back(const Point& point) {
        ++size;
        mean_point_ += point;
        mean_sqr_point_ += enoki::sqr(point);
    }

    auto clear() -> void {
        size = 0;
        mean_point_ = enoki::zero<Point>();
        mean_sqr_point_ = enoki::zero<Point>();
    }
};

} // namespace sdmm
