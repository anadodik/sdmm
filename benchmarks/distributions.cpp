#include <set>
#include <vector>

#include <benchmark/benchmark.h>

#include "jmm/mixture_model.h"

#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

void conditioning_jmm(benchmark::State &state) {
    constexpr static int t_dims = 4;
    constexpr static int t_conditionalDims = 2;
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;
    constexpr static int t_initComponents = 64;
    constexpr static int t_components = 64;
    constexpr static bool USE_BAYESIAN = true;
    using Scalar = float;

    using MM = jmm::MixtureModel<
        t_dims,
        t_components,
        t_conditionalDims,
        Scalar,
        jmm::MultivariateNormal,
        jmm::MultivariateNormal
    >;
    using MMCond = typename MM::ConditionalDistribution;

    auto distribution = MM();
    auto conditional = MMCond();
    distribution.setNComponents(t_components);
    conditional.setNComponents(t_components);
    for(int i = 0; i < t_components; ++i) {
        distribution.weights()[i] = 1.f / 64.f;
        distribution.components()[i].set(
            MM::Vectord::Ones(),
            MM::Matrixd::Identity()
        );
    }
    distribution.configure();

    typename MM::ConditionVectord point({1, 2});
    typename MM::ConditionalVectord query({1, 2});
    Scalar heuristicWeight;
    float pdf_sum = 0;
    float n_repetitions = 0;
    for(auto _ : state) {
        distribution.conditional(point, conditional, heuristicWeight);
        pdf_sum += conditional.pdf(query);
        n_repetitions++;
    }
    // spdlog::info(pdf_sum / n_repetitions);
}

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
    constexpr static size_t NComponents = 64;

    JointSDMM distribution;
    enoki::set_slices(distribution, NComponents);
    distribution = enoki::zero<JointSDMM>(NComponents);
    distribution.tangent_space.mean = enoki::full<sdmm::vector_s_t<JointSDMM>>(1, NComponents);
    distribution.weight.pmf = enoki::full<decltype(distribution.weight.pmf)>(1.f / NComponents, NComponents);
    distribution.cov = enoki::identity<sdmm::matrix_t<JointSDMM>>(NComponents);
    assert(sdmm::prepare(distribution) == true);

    Conditioner conditioner;
    enoki::set_slices(conditioner, enoki::slices(distribution));
    sdmm::prepare(conditioner, distribution);

    Value pdf;
    enoki::set_slices(pdf, enoki::slices(distribution));
    sdmm::vector_s_t<MarginalSDMM> point({1, 2});
    sdmm::vector_s_t<ConditionalSDMM> query({1, 2});
    float pdf_sum = 0;
    float n_repetitions = 0;
    for(auto _ : state) {
        sdmm::create_conditional(conditioner, point);
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(pdf_gaussian),
            conditioner.conditional,
            query,
            pdf
        );
        pdf_sum += enoki::hsum(pdf);
        n_repetitions++;
    }
    // spdlog::info(pdf_sum / n_repetitions);
}

// Register the function as a benchmark
BENCHMARK_TEMPLATE(conditioning, 1);
BENCHMARK_TEMPLATE(conditioning, 2);
BENCHMARK_TEMPLATE(conditioning, 4);
BENCHMARK_TEMPLATE(conditioning, 8);
BENCHMARK_TEMPLATE(conditioning, 16);
BENCHMARK_TEMPLATE(conditioning, 32);

BENCHMARK(conditioning_jmm);

BENCHMARK_MAIN();
