#pragma once

#include <boost/atomic/atomic.hpp>
#include <cstdint>
#include <stdexcept>

#include <enoki/array.h>

namespace sdmm {

struct SpatialStats {
    using Point = sdmm::Vector<float, 3>;

    SpatialStats() {
        clear();
    }
    SpatialStats(SpatialStats& other) = delete;
    SpatialStats& operator=(SpatialStats& other) = delete;
    SpatialStats(SpatialStats&& other) = delete;
    SpatialStats& operator=(SpatialStats&& other) = delete;

    boost::atomic<float> mean_x, mean_y, mean_z;
    boost::atomic<float> mean_sqr_x, mean_sqr_y, mean_sqr_z;
    boost::atomic<uint32_t> size = 0;

    Point mean_point() const {
        return Point(mean_x.load(), mean_y.load(), mean_z.load());
    }
    Point mean_sqr_point() const {
        return Point(mean_sqr_x.load(), mean_sqr_y.load(), mean_sqr_z.load());
    }

    auto push_back(const Point& point) {
        size.fetch_add(1, boost::memory_order_relaxed);
        mean_x.fetch_add(point.coeff(0), boost::memory_order_relaxed);
        mean_y.fetch_add(point.coeff(1), boost::memory_order_relaxed);
        mean_z.fetch_add(point.coeff(2), boost::memory_order_relaxed);

        mean_sqr_x.fetch_add(
            point.coeff(0) * point.coeff(0), boost::memory_order_relaxed);
        mean_sqr_y.fetch_add(
            point.coeff(1) * point.coeff(1), boost::memory_order_relaxed);
        mean_sqr_z.fetch_add(
            point.coeff(2) * point.coeff(2), boost::memory_order_relaxed);
    }

    auto clear() -> void {
        size.store(0);

        mean_x.store(0);
        mean_y.store(0);
        mean_z.store(0);

        mean_sqr_x.store(0);
        mean_sqr_y.store(0);
        mean_sqr_z.store(0);
    }
};

} // namespace sdmm
