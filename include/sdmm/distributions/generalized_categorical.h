#pragma once

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <enoki/array.h>
#include <enoki/dynamic.h>

#include "sdmm/core/utils.h"

namespace sdmm {

template <typename Value_>
struct GeneralizedCategorical {
    using Value = Value_;
    using Scalar = enoki::scalar_t<Value>;
    using Mask = enoki::mask_t<Value>;
    using ValueOuter = sdmm::outer_type_t<Value, Scalar>;
    using BoolOuter = sdmm::outer_type_t<Value, bool>;

    Value pmf;
    Value cdf;

    BoolOuter prepare();


    template <typename RNG>
    auto sample(RNG& rng) -> typename RNG::UInt32;

    ENOKI_STRUCT(GeneralizedCategorical, pmf, cdf);
};

template <typename Value_>
template <typename RNG>
auto GeneralizedCategorical<Value_>::sample(RNG& rng) -> typename RNG::UInt32 {
    auto weight_inv_sample = rng.next_float32();
    using UInt32 = typename RNG::UInt32;
    UInt32 idx = enoki::binary_search(
        0, enoki::slices(cdf) - 1, [&](UInt32 index) {
            return cdf[index] < weight_inv_sample;
        });
    while (idx > 0 && pmf[idx] == 0) {
        --idx;
    }
    return idx;
}

template <
    typename Value_,
    std::enable_if_t<
        enoki::is_array_v<typename GeneralizedCategorical<Value_>::BoolOuter>,
        int> = 0>
[[nodiscard]] auto is_valid(const GeneralizedCategorical<Value_>& categorical) ->
    typename GeneralizedCategorical<Value_>::BoolOuter {
    using GeneralizedCategoricalV = GeneralizedCategorical<Value_>;
    const typename GeneralizedCategoricalV::Mask zero_values =
        enoki::neq(categorical.pmf, 0.f);
    typename GeneralizedCategoricalV::BoolOuter valid_pmf = false;
    for (size_t i = 0;
         i < enoki::array_size_v<typename GeneralizedCategoricalV::BoolOuter>;
         ++i) {
        valid_pmf.coeff(i) = enoki::any(zero_values.coeff(i));
    }
    if (!enoki::all(valid_pmf)) {
        enoki::bool_array_t<typename GeneralizedCategoricalV::BoolOuter> bool_array =
            valid_pmf;
        spdlog::warn("GeneralizedCategorical::is_valid()={}.", bool_array);
    }
    return valid_pmf;
}

template <
    typename Value_,
    std::enable_if_t<
        !enoki::is_array_v<typename GeneralizedCategorical<Value_>::BoolOuter>,
        int> = 0>
[[nodiscard]] auto is_valid(const GeneralizedCategorical<Value_>& categorical) ->
    typename GeneralizedCategorical<Value_>::BoolOuter {
    using GeneralizedCategoricalV = GeneralizedCategorical<Value_>;
    const typename GeneralizedCategoricalV::Mask zero_values =
        enoki::neq(categorical.pmf, 0.f);
    typename GeneralizedCategoricalV::BoolOuter valid_pmf = false;
    valid_pmf = enoki::any(zero_values);
    if (!enoki::all(valid_pmf)) {
        enoki::bool_array_t<typename GeneralizedCategoricalV::BoolOuter> bool_array =
            valid_pmf;
        spdlog::warn("GeneralizedCategorical::is_valid()={}.", bool_array);
    }
    return valid_pmf;
}


template <typename Value_>
[[nodiscard]] auto GeneralizedCategorical<Value_>::prepare() -> BoolOuter {
    size_t n_slices = enoki::slices(pmf);
    if (enoki::slices(cdf) != n_slices) {
        enoki::set_slices(cdf, n_slices);
    }
    enoki::slice(cdf, 0) = enoki::slice(pmf, 0);
    for (size_t i = 1; i < n_slices; ++i) {
        enoki::slice(cdf, i) = enoki::slice(cdf, i - 1) + enoki::slice(pmf, i);
    }

    ValueOuter pmf_sum = enoki::slice(cdf, n_slices - 1);
    BoolOuter is_valid = pmf_sum > 1e-20f;
    if (!enoki::any(is_valid)) {
        return is_valid;
    }
    ValueOuter inv_normalizer = 1 / enoki::select(is_valid, pmf_sum, 1.f);

    // This can be further optimized by
    // only iterating over cdfs which have non-zero sums.
    auto normalize = [inv_normalizer](auto&& pmf, auto&& cdf) {
        cdf *= inv_normalizer;
        pmf *= inv_normalizer;
    };

    enoki::vectorize(normalize, pmf, cdf);

    return is_valid;
}

} // namespace sdmm

ENOKI_STRUCT_SUPPORT(sdmm::GeneralizedCategorical, pmf, cdf);
