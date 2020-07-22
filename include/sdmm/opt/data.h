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

template<typename SDMM_>
struct Data {
    using SDMM = SDMM_;
    using Scalar = typename SDMM::Scalar;
    using Embedded = sdmm::embedded_t<SDMM>;
    using Matrix = sdmm::matrix_t<SDMM>;
    using Normal = sdmm::Vector<Scalar, 3>;

    using ScalarS = enoki::scalar_t<Scalar>;
    using EmbeddedS = sdmm::embedded_s_t<SDMM>;
    using NormalS = sdmm::Vector<ScalarS, 3>;

    Embedded point;
    Normal normal;
    Scalar weight;
    Scalar heuristic_pdf;

    EmbeddedS mean_point = 0;
    EmbeddedS mean_sqr_point = 0;

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

        if(weight_ == 0) {
            return;
        }
        
        mean_point += point_;
        mean_sqr_point += enoki::sqr(point_);

        if(size >= enoki::slices(point)) {
            // if(capacity > enoki::slices(point)) {
            //     enoki::set_slices(*this, capacity);
            // } else {
                throw std::runtime_error("Data full.\n");
                return;
            // }
        }

        enoki::slice(point, size) = point_;
        enoki::slice(normal, size) = normal_;
        enoki::slice(weight, size) = weight_;
        enoki::slice(heuristic_pdf, size) = heuristic_pdf_;
        ++size;
    }

    auto clear() -> void {
        size = 0;
        mean_point = 0;
        mean_sqr_point = 0;
    }

    auto reserve(uint32_t new_capacity) -> void {
        capacity = new_capacity; 
        enoki::set_slices(*this, capacity);
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
