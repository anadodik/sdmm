#pragma once

#include <enoki/array.h>

#include "sdmm/core/utils.h"
#include "sdmm/distributions/sdmm.h"

namespace sdmm {

template<typename SDMM_>
struct Data {
    using SDMM = SDMM_;
    using Scalar = typename SDMM::Scalar;
    using Embedded = sdmm::embedded_t<SDMM>;
    using Matrix = sdmm::matrix_t<SDMM>;

    using ScalarS = enoki::scalar_t<Scalar>;

    Embedded point;
    Scalar weight;

    auto remove_non_finite() -> void;

    ENOKI_STRUCT(Data, point, weight);
};

template<typename SDMM_>
auto Data<SDMM_>::remove_non_finite() -> void {
    weight = enoki::select(enoki::isfinite(weight), weight, ScalarS(0));
}

}

ENOKI_STRUCT_SUPPORT(sdmm::Data, point, weight);
