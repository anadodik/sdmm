#pragma once

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <enoki/array.h>
#include <enoki/dynamic.h>

#include "sdmm/core/utils.h"

namespace sdmm {

template<typename Value_>
struct Categorical {
    using Value = Value_;
    using Scalar = enoki::scalar_t<Value>;
    using Mask = enoki::mask_t<Value>;
    using ValueOuter = sdmm::outer_type_t<Value, Scalar>;
    using BoolOuter = sdmm::outer_type_t<Value, bool>;

    Value pmf;
    Value cdf;

    BoolOuter prepare();
    void normalize_cdf(const ValueOuter& inv_normalizer);
    BoolOuter is_valid();

    ENOKI_STRUCT(Categorical, pmf, cdf);
};

template<typename Value_>
[[nodiscard]] auto Categorical<Value_>::is_valid() -> BoolOuter {
    Mask zero_values = enoki::neq(pmf, 0.f);
    BoolOuter valid_pmf = false;
    for(size_t i = 0; i < enoki::array_size_v<BoolOuter>; ++i) {
        coeff_safe(valid_pmf, i) = enoki::any(coeff_safe(zero_values, i));
    }
    if(!enoki::all(valid_pmf)) {
        enoki::bool_array_t<BoolOuter> bool_array = valid_pmf;
        spdlog::warn("Categorical::is_valid()={}.", bool_array);
    }
    return valid_pmf;
}

template<typename Value_>
auto Categorical<Value_>::normalize_cdf(
    const ValueOuter& inv_normalizer
) -> void {
    cdf *= inv_normalizer;
}

template<typename Value_>
[[nodiscard]] auto Categorical<Value_>::prepare() -> BoolOuter {
    // BoolOuter valid = enoki::vectorize(
    //     VECTORIZE_WRAP_MEMBER(is_valid),
    //     *this
    // );
    BoolOuter valid = is_valid();
    if(enoki::none(valid)) {
        return valid;
    }

    size_t n_slices = enoki::slices(pmf);
    enoki::slice(cdf, 0) = enoki::slice(pmf, 0);
    for(size_t i = 1; i < n_slices; ++i) {
        enoki::slice(cdf, i) = enoki::slice(cdf, i - 1) + enoki::slice(pmf, i);
    }

    ValueOuter cdf_sum = enoki::slice(cdf, n_slices - 1);
    ValueOuter inv_normalizer = 1 / enoki::select(cdf_sum > 0.f, cdf_sum, 1.f);

    // This can be further optimized by
    // only iterating over cdfs which have non-zero sums.
    // normalize_cdf(inv_normalizer);
    enoki::vectorize(
        VECTORIZE_WRAP_MEMBER(normalize_cdf),
        *this,
        inv_normalizer
    );

    return valid;
}

}

ENOKI_STRUCT_SUPPORT(sdmm::Categorical, pmf, cdf);
