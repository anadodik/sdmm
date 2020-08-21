#include <set>
#include <vector>

#include <enoki/random.h>

#include <benchmark/benchmark.h>

#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

#include "sdmm/spaces/euclidian.h"
#include "sdmm/spaces/directional.h"
#include "sdmm/spaces/spatio_directional.h"

using Scalar = float;

template<typename Vector, typename RNG>
auto sample_vector(RNG& rng, Vector& sample) -> void {
    for(size_t dim_i = 0; dim_i < Vector::Size; ++dim_i) {
        sample.coeff(dim_i) = rng.next_float32();
    }
}

template<size_t PacketSize>
void sampling(benchmark::State &state) {
    constexpr static size_t JointSize = 5;

    using Packet = enoki::Packet<Scalar, PacketSize>;
    using Value = enoki::DynamicArray<Packet>;
    using JointTangentSpace = sdmm::SpatioDirectionalTangentSpace<
        sdmm::Vector<Value, JointSize + 1>, sdmm::Vector<Value, JointSize>
    >;
    using JointSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, JointSize>, JointTangentSpace
    >;
    using JointCov = sdmm::matrix_t<JointSDMM>;
    using JointEmbedded = sdmm::embedded_t<JointSDMM>;

    constexpr static size_t NComponents = 64;
    JointSDMM distribution;
    enoki::set_slices(distribution, NComponents);
    distribution = enoki::zero<JointSDMM>(NComponents);
    distribution.tangent_space.set_mean(
        sdmm::full_inner<JointEmbedded, Value, NComponents>(
            1, 1, 1, 1, 0, 0
        )
    );
    distribution.weight.pmf =
        enoki::full<decltype(distribution.weight.pmf)>(1.f / NComponents, NComponents);
    distribution.cov = enoki::identity<JointCov>(NComponents);
    bool prepare_success = sdmm::prepare_vectorized(distribution);
    assert(prepare_success == true);

    constexpr static int NSamples = 4;
    using RNG = enoki::PCG32<Value, NSamples>;
    RNG rng;

    Value pdf;
    sdmm::replace_embedded_t<JointSDMM, Value> sample;
    sdmm::replace_tangent_t<JointSDMM, Value> tangent_sample;
    for(auto _ : state) {
        // FIXME
        // distribution.sample(rng, sample, pdf, tangent_sample);
        spdlog::info("tangent_sample: {}", tangent_sample);
    }

    using UIntP = enoki::Packet<uint32_t, PacketSize>;
    Packet value = enoki::arange<Packet>() / PacketSize;
    auto j = enoki::binary_search(
        0,
        enoki::slices(distribution.weight.cdf),
        [&](UIntP index) {
            return enoki::gather<Packet>(distribution.weight.cdf, index) < value;
        }
    );
    spdlog::info("pmf: {}", distribution.weight.pmf);
    spdlog::info("cdf: {}", distribution.weight.cdf);
    spdlog::info("sampled: {}", j);
}

BENCHMARK_TEMPLATE(sampling, 32);
