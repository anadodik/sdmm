#pragma once

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

    Embedded point;
    Normal normal;
    Scalar weight;
    Scalar heuristic_pdf;

    EmbeddedStats mean_point = 0;
    EmbeddedStats mean_sqr_point = 0;
    PositionStats min_position = 0;
    PositionStats max_position = 0;
    uint32_t stats_size = 0;

    uint32_t size = 0;
    uint32_t capacity = 0;

    auto remove_non_finite() -> void;

    template<typename DataSlice>
    auto push_back(DataSlice&& other) -> void {
        push_back(
            other.point,
            other.normal,
            other.weight,
            other.heuristic_pdf
        );
    }

    template<typename EmbeddedIn, typename NormalIn, typename ScalarIn>
    auto push_back(
        const EmbeddedIn& point_,
        const NormalIn& normal_,
        const ScalarIn& weight_,
        const ScalarIn& heuristic_pdf_
    ) -> void {
        if(!is_valid_sample(weight_)) {
            return;
        }

        if(size >= enoki::slices(point)) {
            // if(capacity > enoki::slices(point)) {
            //     enoki::set_slices(*this, capacity);
            // } else {
                throw std::runtime_error("Data full.\n");
                return;
            // }
        }
        
        mean_point += point_;
        mean_sqr_point += enoki::sqr(point_);
        enoki::expr_t<EmbeddedIn> point_copy = point_;
        min_position = enoki::min(min_position, enoki::head<3>(point_copy));
        max_position = enoki::max(max_position, enoki::head<3>(point_copy));
        ++stats_size;

        enoki::slice(point, size) = point_;
        enoki::slice(normal, size) = normal_;
        enoki::slice(weight, size) = weight_;
        enoki::slice(heuristic_pdf, size) = heuristic_pdf_;
        ++size;
    }

    auto clear() -> void {
        size = 0;
    }

    auto clear_stats() -> void {
        mean_point = enoki::zero<EmbeddedStats>();
        mean_sqr_point = enoki::zero<EmbeddedStats>();
        min_position = enoki::full<PositionStats>(std::numeric_limits<float>::infinity());
        max_position = enoki::full<PositionStats>(-std::numeric_limits<float>::infinity());
        stats_size = 0;
    }

    auto reserve(uint32_t new_capacity) -> void {
        capacity = new_capacity; 
        enoki::set_slices(*this, capacity);
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

    ENOKI_STRUCT(Data, point, normal, weight, heuristic_pdf);
};

template<typename SDMM_>
auto Data<SDMM_>::remove_non_finite() -> void {
    weight = enoki::select(enoki::isfinite(weight) || weight < 0, weight, ScalarS(0));
}

}

ENOKI_STRUCT_SUPPORT(sdmm::Data, point, normal, weight, heuristic_pdf);
