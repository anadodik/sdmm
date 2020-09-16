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

    ENOKI_STRUCT(Categorical, pmf, cdf);
};

template<typename Value_, std::enable_if_t<enoki::is_array_v<typename Categorical<Value_>::BoolOuter>, int> = 0>
[[nodiscard]] auto is_valid(const Categorical<Value_>& categorical) -> typename Categorical<Value_>::BoolOuter {
    using CategoricalV = Categorical<Value_>;
    const typename CategoricalV::Mask zero_values = enoki::neq(categorical.pmf, 0.f);
    typename CategoricalV::BoolOuter valid_pmf = false;
    for(size_t i = 0; i < enoki::array_size_v<typename CategoricalV::BoolOuter>; ++i) {
        valid_pmf.coeff(i) = enoki::any(zero_values.coeff(i));
    }
    if(!enoki::all(valid_pmf)) {
        enoki::bool_array_t<typename CategoricalV::BoolOuter> bool_array = valid_pmf;
        spdlog::warn("Categorical::is_valid()={}.", bool_array);
    }
    return valid_pmf;
}

template<typename Value_, std::enable_if_t<!enoki::is_array_v<typename Categorical<Value_>::BoolOuter>, int> = 0>
[[nodiscard]] auto is_valid(const Categorical<Value_>& categorical) -> typename Categorical<Value_>::BoolOuter {
    using CategoricalV = Categorical<Value_>;
    const typename CategoricalV::Mask zero_values = enoki::neq(categorical.pmf, 0.f);
    typename CategoricalV::BoolOuter valid_pmf = false;
    valid_pmf = enoki::any(zero_values);
    if(!enoki::all(valid_pmf)) {
        enoki::bool_array_t<typename CategoricalV::BoolOuter> bool_array = valid_pmf;
        spdlog::warn("Categorical::is_valid()={}.", bool_array);
    }
    return valid_pmf;
}

template<typename Value_>
[[nodiscard]] auto Categorical<Value_>::prepare() -> BoolOuter {
    size_t n_slices = enoki::slices(pmf);
    if(enoki::slices(cdf) != n_slices) {
        enoki::set_slices(cdf, n_slices);
    }
    enoki::slice(cdf, 0) = enoki::slice(pmf, 0);
    for(size_t i = 1; i < n_slices; ++i) {
        enoki::slice(cdf, i) = enoki::slice(cdf, i - 1) + enoki::slice(pmf, i);
    }

    ValueOuter pmf_sum = enoki::slice(cdf, n_slices - 1);
    bool is_valid = enoki::all(pmf_sum > 1e-20f);
    if(!is_valid) {
        return is_valid;
    }
    ValueOuter inv_normalizer = 1 / enoki::select(is_valid, pmf_sum, 1.f);

    // This can be further optimized by
    // only iterating over cdfs which have non-zero sums.
    auto normalize = [inv_normalizer](auto&& pmf, auto&& cdf) {
        cdf *= inv_normalizer;
        pmf *= inv_normalizer;
    };

    enoki::vectorize(
        normalize, pmf, cdf
    );

    return is_valid;
}

}

ENOKI_STRUCT_SUPPORT(sdmm::Categorical, pmf, cdf);
