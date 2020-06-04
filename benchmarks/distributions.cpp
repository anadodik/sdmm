#include <set>
#include <vector>

#include <benchmark/benchmark.h>

#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

template<size_t PacketSize>
void conditioning(benchmark::State &state) {
    using Packet = enoki::Packet<float, PacketSize>;
    using Value = enoki::DynamicArray<Packet>;
    using JointTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, 4>, sdmm::Vector<Value, 4>
    >;
    using JointSDMM = sdmm::SDMM<
        sdmm::Vector<Value, 4>, sdmm::Matrix<Value, 4>, JointTangentSpace
    >;
    using MarginalTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, 2>, sdmm::Vector<Value, 2>
    >;
    using MarginalSDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, MarginalTangentSpace
    >;
    using ConditionalTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, 2>, sdmm::Vector<Value, 2>
    >;
    using ConditionalSDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, ConditionalTangentSpace
    >;

    using Conditioner = sdmm::SDMMConditioner<
        JointSDMM, MarginalSDMM, ConditionalSDMM
    >;
    JointSDMM distribution;
    enoki::set_slices(distribution, 64);
    distribution = enoki::zero<JointSDMM>(64);

    Conditioner conditioner;
    enoki::set_slices(conditioner, enoki::slices(distribution));
    Value pdf;
    enoki::set_slices(pdf, enoki::slices(distribution));
    sdmm::vector_s_t<MarginalSDMM> point({1, 2});
    sdmm::vector_s_t<ConditionalSDMM> query({1, 2});
    for(auto _ : state) {
        sdmm::prepare(conditioner, distribution);
        sdmm::create_conditional(conditioner, point);
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(pdf_gaussian),
            conditioner.conditional,
            query,
            pdf
        );
    }
}

// Register the function as a benchmark
BENCHMARK_TEMPLATE(conditioning, 1);
BENCHMARK_TEMPLATE(conditioning, 2);
BENCHMARK_TEMPLATE(conditioning, 4);
BENCHMARK_TEMPLATE(conditioning, 8);
BENCHMARK_TEMPLATE(conditioning, 16);
BENCHMARK_TEMPLATE(conditioning, 32);

BENCHMARK_MAIN();
