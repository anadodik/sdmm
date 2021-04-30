#include <set>
#include <vector>

#include <benchmark/benchmark.h>

#include "sdmm/core/utils.h"

#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

#include "sdmm/opt/em.h"

#include "sdmm/spaces/directional.h"
#include "sdmm/spaces/euclidian.h"
#include "sdmm/spaces/spatio_directional.h"

#if COMPARISON == 1
#include "jmm/mixture_model.h"
#include "jmm/opt/stepwise_tangent.h"
#endif

using Scalar = float;

#if COMPARISON == 1
void optimize_jmm(benchmark::State& state) {
    constexpr static int t_dims = 6;
    constexpr static int t_conditionalDims = 3;
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
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal>;
    using StepwiseEMType = jmm::StepwiseTangentEM<
        t_dims,
        t_components,
        t_conditionalDims,
        Scalar,
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal>;
    using SamplesType = jmm::Samples<t_dims, Scalar>;
    constexpr static size_t NPoints = 1000000;

    StepwiseEMType optimizer(
        0.5, Eigen::Matrix<Scalar, 5, 1>::Identity() * 1e-5, 1e-3);
    SamplesType samples;
    samples.reserve(NPoints);
    samples.setSize(NPoints);
    for (size_t sample_i = 0; sample_i < NPoints; ++sample_i) {
        samples.samples.col(sample_i) << 1, 1, 1, 1, 0, 0;
        samples.weights(sample_i) = 1;
        samples.isDiffuse(sample_i) = true;
        samples.heuristicPdfs(sample_i) = 0.5;
    }

    auto distribution = MM();
    distribution.setNComponents(t_components);
    MM::Vectord mean;
    mean << 1, 1, 1, 1, 0, 0;
    MM::Matrixd cov_prior;
    cov_prior << 1e-4, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0, 1e-4, 0, 0, 0, 0, 0,
        1e-5, 0, 0, 0, 0, 0, 1e-5;
    for (int i = 0; i < t_components; ++i) {
        distribution.weights()[i] = 1.f / 64.f;
        distribution.components()[i].set(mean, MM::Matrixd::Identity());
        optimizer.getBPriors()[i] = cov_prior;
    }
    distribution.configure();

    for (auto _ : state) {
        Scalar max_error;
        optimizer.optimize(distribution, samples, max_error);
    }
}
BENCHMARK(optimize_jmm)->Unit(benchmark::kMillisecond);
#endif // COMPARISON == 1

template <size_t PacketSize>
void optimize(benchmark::State& state) {
    constexpr static size_t JointSize = 5;

    using Packet = enoki::Packet<Scalar, PacketSize>;
    using Value = enoki::DynamicArray<Packet>;
    using JointTangentSpace = sdmm::SpatioDirectionalTangentSpace<
        sdmm::Vector<Value, JointSize + 1>,
        sdmm::Vector<Value, JointSize>>;
    using JointSDMM =
        sdmm::SDMM<sdmm::Matrix<Value, JointSize>, JointTangentSpace>;
    using JointCov = sdmm::matrix_t<JointSDMM>;
    using JointEmbedded = sdmm::embedded_t<JointSDMM>;

    constexpr static size_t NComponents = 64;
    JointSDMM distribution;
    enoki::set_slices(distribution, NComponents);
    distribution = enoki::zero<JointSDMM>(NComponents);
    distribution.tangent_space.set_mean(
        sdmm::full_inner<JointEmbedded, Value, NComponents>(1, 1, 1, 1, 0, 0));
    distribution.weight.pmf = enoki::full<decltype(distribution.weight.pmf)>(
        1.f / NComponents, NComponents);
    distribution.cov = enoki::identity<JointCov>(NComponents);
    assert(sdmm::prepare_vectorized(distribution) == true);

    using Data = sdmm::Data<JointSDMM>;
    constexpr static size_t NPoints = 1000000;
    Data data;
    enoki::set_slices(data, NPoints);
    data.weight = enoki::full<Value>(1, NPoints);
    data.point =
        sdmm::full_inner<JointEmbedded, Value, NPoints>(1, 1, 1, 1, 0, 0);

    using EM = sdmm::EM<JointSDMM>;
    EM em;
    em = enoki::zero<EM>(NComponents);
    Scalar weight_prior = 1.f / NComponents;
    Scalar cov_prior_strength = 5.f / NComponents;
    JointCov cov_prior = sdmm::full_inner<JointCov, Value, NComponents>(
        2e-3,
        0,
        0,
        0,
        0,
        0,
        2e-3,
        0,
        0,
        0,
        0,
        0,
        2e-3,
        0,
        0,
        0,
        0,
        0,
        2e-4,
        0,
        0,
        0,
        0,
        0,
        2e-4);
    em.set_priors(weight_prior, cov_prior_strength, cov_prior);
    for (auto _ : state) {
        em.compute_stats_model_parallel(distribution, data);
        em.interpolate_stats();
        assert(em.normalize_stats(data));
        enoki::vectorize_safe(
            VECTORIZE_WRAP(sdmm::update_model), distribution, em);
        assert(sdmm::prepare_vectorized(distribution) == true);
    }
}

// BENCHMARK_TEMPLATE(optimize, 1)->Unit(benchmark::kMillisecond);
// BENCHMARK_TEMPLATE(optimize, 4)->Unit(benchmark::kMillisecond);
// BENCHMARK_TEMPLATE(optimize, 8)->Unit(benchmark::kMillisecond);
// BENCHMARK_TEMPLATE(optimize, 16)->Unit(benchmark::kMillisecond);
// BENCHMARK_TEMPLATE(optimize, 32)->Unit(benchmark::kMillisecond);
