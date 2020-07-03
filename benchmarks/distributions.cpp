#include <set>
#include <vector>

#include <benchmark/benchmark.h>

#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

#if COMPARISON == 1
#include "jmm/mixture_model.h"
#endif

using Scalar = float;

#if COMPARISON == 1
void conditioning_jmm(benchmark::State &state) {
    constexpr static int t_dims = 5;
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

void conditioning_jmm_spatio_directional(benchmark::State &state) {
    constexpr static int t_dims = 6;
    constexpr static int t_conditionalDims = 3;
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;
    constexpr static int t_initComponents = 64;
    constexpr static int t_components = 64;
    constexpr static bool USE_BAYESIAN = true;

    using MM = jmm::MixtureModel<
        t_dims,
        t_components,
        t_conditionalDims,
        Scalar,
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal
    >;
    using MMCond = typename MM::ConditionalDistribution;

    auto distribution = MM();
    auto conditional = MMCond();
    distribution.setNComponents(t_components);
    conditional.setNComponents(t_components);
    MM::Vectord mean; mean << 1, 1, 1, 1, 0, 0;
    for(int i = 0; i < t_components; ++i) {
        distribution.weights()[i] = 1.f / 64.f;
        distribution.components()[i].set(
            mean,
            MM::Matrixd::Identity()
        );
    }
    distribution.configure();

    typename MM::ConditionVectord point({1, 1, 1});
    Scalar heuristicWeight;
    float pdf;
    float pdf_sum = 0;
    float mean_n_comp = 0;
    float n_repetitions = 0;
    int query_i = 0;
    for(auto _ : state) {
        distribution.conditional(point, conditional, heuristicWeight);
        typename MM::ConditionalVectord query({query_i % 3, (query_i + 1) % 3, (query_i + 2) % 3});
        // distribution.conditional(point, conditional, heuristicWeight);
        // mean_n_comp += conditional.nComponents();
        pdf = conditional.pdf(query);
        pdf_sum += pdf;
        benchmark::DoNotOptimize(n_repetitions++);
        ++query_i;
    }
    spdlog::info("pdf: {}", pdf);
    spdlog::info("mean pdf: {}", pdf_sum / n_repetitions);
    spdlog::info("pdf_sum: {}", pdf_sum);
}

// BENCHMARK(conditioning_jmm);
BENCHMARK(conditioning_jmm_spatio_directional);
#endif

template<size_t PacketSize>
void conditioning(benchmark::State &state) {
    constexpr static size_t JointSize = 5;
    constexpr static size_t MarginalSize = 3;
    constexpr static size_t ConditionalSize = 2;
    static_assert(JointSize == MarginalSize + ConditionalSize);

    using Packet = enoki::Packet<Scalar, PacketSize>;
    using Value = enoki::DynamicArray<Packet>;
    using JointTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, JointSize>, sdmm::Vector<Value, JointSize>
    >;
    using JointSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, JointSize>, JointTangentSpace
    >;
    using MarginalTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, MarginalSize>, sdmm::Vector<Value, MarginalSize>
    >;
    using MarginalSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, MarginalSize>, MarginalTangentSpace
    >;
    using ConditionalTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, ConditionalSize>, sdmm::Vector<Value, ConditionalSize>
    >;
    using ConditionalSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, ConditionalSize>, ConditionalTangentSpace
    >;

    using Conditioner = sdmm::SDMMConditioner<
        JointSDMM, MarginalSDMM, ConditionalSDMM
    >;
    constexpr static size_t NComponents = 64;

    JointSDMM distribution;
    enoki::set_slices(distribution, NComponents);
    distribution = enoki::zero<JointSDMM>(NComponents);
    distribution.tangent_space.mean = enoki::full<sdmm::embedded_s_t<JointSDMM>>(1, NComponents);
    distribution.weight.pmf = enoki::full<decltype(distribution.weight.pmf)>(1.f / NComponents, NComponents);
    distribution.cov = enoki::identity<sdmm::matrix_t<JointSDMM>>(NComponents);
    assert(sdmm::prepare(distribution) == true);

    Conditioner conditioner;
    enoki::set_slices(conditioner, enoki::slices(distribution));
    sdmm::prepare(conditioner, distribution);

    Value pdf;
    enoki::set_slices(pdf, enoki::slices(distribution));
    sdmm::embedded_s_t<MarginalSDMM> point({1, 2, 2});
    sdmm::embedded_s_t<ConditionalSDMM> query({1, 2});
    float pdf_sum = 0;
    float n_repetitions = 0;
    sdmm::create_conditional(conditioner, point);
    for(auto _ : state) {
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

template<size_t PacketSize>
void conditioning_spatio_directional(benchmark::State &state) {
    constexpr static size_t JointSize = 5;
    constexpr static size_t MarginalSize = 3;
    constexpr static size_t ConditionalSize = 2;
    static_assert(JointSize == MarginalSize + ConditionalSize);

    using Packet = enoki::Packet<Scalar, PacketSize>;
    using Value = enoki::DynamicArray<Packet>;
    using JointTangentSpace = sdmm::SpatioDirectionalTangentSpace<
        sdmm::Vector<Value, JointSize + 1>, sdmm::Vector<Value, JointSize>
    >;
    using JointSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, JointSize>, JointTangentSpace
    >;
    using MarginalTangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, MarginalSize>, sdmm::Vector<Value, MarginalSize>
    >;
    using MarginalSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, MarginalSize>, MarginalTangentSpace
    >;
    using ConditionalTangentSpace = sdmm::DirectionalTangentSpace<
        sdmm::Vector<Value, ConditionalSize + 1>, sdmm::Vector<Value, ConditionalSize>
    >;
    using ConditionalSDMM = sdmm::SDMM<
        sdmm::Matrix<Value, ConditionalSize>, ConditionalTangentSpace
    >;

    using Conditioner = sdmm::SDMMConditioner<
        JointSDMM, MarginalSDMM, ConditionalSDMM
    >;
    constexpr static size_t NComponents = 64;

    JointSDMM distribution;
    enoki::set_slices(distribution, NComponents);
    distribution = enoki::zero<JointSDMM>(NComponents);
    distribution.tangent_space.set_mean(
        sdmm::embedded_t<JointSDMM>{
        enoki::full<Value>(1, NComponents),
        enoki::full<Value>(1, NComponents),
        enoki::full<Value>(1, NComponents),
        enoki::full<Value>(1, NComponents),
        enoki::full<Value>(0, NComponents),
        enoki::full<Value>(0, NComponents)
        }
    );
    distribution.weight.pmf =
        enoki::full<decltype(distribution.weight.pmf)>(1.f / NComponents, NComponents);
    distribution.cov = enoki::identity<sdmm::matrix_t<JointSDMM>>(NComponents);
    assert(sdmm::prepare(distribution) == true);

    Conditioner conditioner;
    enoki::set_slices(conditioner, enoki::slices(distribution));
    sdmm::prepare(conditioner, distribution);

    Value pdf;
    enoki::set_slices(pdf, enoki::slices(distribution));
    sdmm::embedded_s_t<MarginalSDMM> point({1, 1, 1});
    sdmm::embedded_s_t<ConditionalSDMM> query({1, 0, 0});
    float pdf_hsum = 0;
    float pdf_sum = 0;
    float n_repetitions = 0;
    int query_i = 0;
    for(auto _ : state) {
        sdmm::create_conditional(conditioner, point);
        sdmm::embedded_s_t<ConditionalSDMM>{Scalar(query_i % 3), Scalar((query_i + 1) % 3), Scalar((query_i + 2) % 3)};
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(posterior),
            conditioner.conditional,
            query,
            pdf
        );
        pdf_hsum = enoki::hsum_nested(pdf);

        benchmark::DoNotOptimize(pdf_sum += pdf_hsum);
        benchmark::DoNotOptimize(n_repetitions++);
        ++query_i;
    }
}

// Register the function as a benchmark
// BENCHMARK_TEMPLATE(conditioning, 1);
// BENCHMARK_TEMPLATE(conditioning, 2);
// BENCHMARK_TEMPLATE(conditioning, 4);
// BENCHMARK_TEMPLATE(conditioning, 8);
// BENCHMARK_TEMPLATE(conditioning, 16);
// BENCHMARK_TEMPLATE(conditioning, 32);

BENCHMARK_TEMPLATE(conditioning_spatio_directional, 1);
BENCHMARK_TEMPLATE(conditioning_spatio_directional, 4);
BENCHMARK_TEMPLATE(conditioning_spatio_directional, 8);
BENCHMARK_TEMPLATE(conditioning_spatio_directional, 16);
BENCHMARK_TEMPLATE(conditioning_spatio_directional, 32);

BENCHMARK_MAIN();
