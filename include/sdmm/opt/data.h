#pragma once

#include <atomic>
#include <cstdint>
#include <stdexcept>

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/distributions/sdmm.h"

namespace sdmm {

template<typename T>
using normal_t = typename T::Normal;

template<typename T>
using normal_s_t = typename T::NormalS;

template<typename Scalar>
auto is_valid_sample(Scalar&& weight) -> bool {
    return (std::isfinite(weight) && weight >= 1e-8);
}

template<typename SDMM_>
struct Data {
    Data(const Data& other) : capacity(other.capacity), size(other.size) {
        reserve(capacity);
    }
    Data(Data&& other) = default;
    Data& operator=(Data&& other) = default;
    virtual ~Data() = default;

    using SDMM = SDMM_;
    using Scalar = typename SDMM::Scalar;
    using Embedded = sdmm::embedded_t<SDMM>;
    using Matrix = sdmm::matrix_t<SDMM>;
    using Normal = sdmm::Vector<Scalar, 3>;
    using Position = sdmm::Vector<Scalar, 3>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using EmbeddedS = sdmm::embedded_s_t<SDMM>;
    using NormalS = sdmm::Vector<ScalarS, 3>;
    using PositionS = sdmm::Vector<ScalarS, 3>;

    using EmbeddedStats = enoki::replace_scalar_t<EmbeddedS, double>;
    using PositionStats = PositionS; // enoki::replace_scalar_t<PositionS, double>;

    // If adding a new data-entry, remember to update reserve.
    Embedded point;
    Normal normal;
    Scalar weight;

    EmbeddedStats mean_point = 0;
    EmbeddedStats mean_sqr_point = 0;
    uint32_t stats_size = 0;

    uint32_t size = 0;
    uint32_t capacity = 0;

    auto remove_non_finite() -> void;

    template<typename DataSlice>
    auto push_back(DataSlice&& other) -> void {
        push_back(
            other.point,
            other.normal,
            other.weight
        );
    }

    template<typename EmbeddedIn, typename NormalIn, typename ScalarIn>
    auto push_back(
        const EmbeddedIn& point_,
        const NormalIn& normal_,
        const ScalarIn& weight_
    ) -> void {
        if(!is_valid_sample(weight_)) {
            return;
        }

        uint32_t idx = size++;

        if(idx >= enoki::slices(point)) {
            // if(capacity > enoki::slices(point)) {
            //     enoki::set_slices(*this, capacity);
            // } else {
                throw std::runtime_error("Data full.\n");
                return;
            // }
        }
        enoki::slice(point, idx) = point_;
        enoki::slice(normal, idx) = normal_;
        enoki::slice(weight, idx) = weight_;

        ++stats_size;
        mean_point += point_;
        mean_sqr_point += enoki::sqr(point_);
    }

    auto clear() -> void {
        size = 0;
    }

    auto clear_stats() -> void {
        mean_point = enoki::zero<EmbeddedStats>();
        mean_sqr_point = enoki::zero<EmbeddedStats>();
        stats_size = 0;
    }

    auto reserve(uint32_t new_capacity) -> void {
        // spdlog::info("new_capacity={}", new_capacity);
        capacity = new_capacity;
        // enoki::set_slices(*this, capacity);
        enoki::set_slices(point, capacity);
        enoki::set_slices(normal, capacity);
        enoki::set_slices(weight, capacity);
        clear();
        clear_stats();
    }

    auto sum_weights() -> ScalarS {
        using Float = enoki::Packet<ScalarS, 8>;
        using Index = enoki::Packet<uint32_t, 8>;
        Float packet_sum(0);
        for (auto [index, mask] : enoki::range<Index>(size)) {
            packet_sum += select(
                mask,
                enoki::gather<Float>(weight, index),
                Index(0)
            );
        }
        return enoki::hsum(packet_sum);
    }

    ENOKI_STRUCT(Data, point, normal, weight);
};

template<typename SDMM_>
auto Data<SDMM_>::remove_non_finite() -> void {
    weight = enoki::select(enoki::isfinite(weight) || weight < 0, weight, ScalarS(0));
}

}

ENOKI_STRUCT_SUPPORT(sdmm::Data, point, normal, weight);
