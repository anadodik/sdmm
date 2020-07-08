#include <doctest/doctest.h>

#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/dynamic.h>

#include "sdmm/distributions/categorical.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

#include "utils.h"

#if COMPARISON == 1
#include "jmm/mixture_model.h"
#endif

using Scalar = float;

template<typename JMM, typename SDMM, size_t... Indices>
void copy_means(JMM& jmm, SDMM& sdmm, std::index_sequence<Indices...>) {
    size_t NComponents = jmm.nComponents();
    for(size_t component_i = 0; component_i < NComponents; ++component_i) {
        enoki::slice(sdmm.tangent_space, component_i).set_mean(
            sdmm::embedded_s_t<SDMM>(
                jmm.components()[component_i].mean()(Indices)...
            )
        );
    }
}

template<typename JMM, typename SDMM, size_t... Indices>
void copy_covs(JMM& jmm, SDMM& sdmm, std::index_sequence<Indices...>) {
    size_t NComponents = jmm.nComponents();
    for(size_t component_i = 0; component_i < NComponents; ++component_i) {
        enoki::slice(sdmm.cov, component_i) = sdmm::matrix_s_t<SDMM>(
            jmm.components()[component_i].cov()(Indices)...
        );
    }
}

template<typename JMM, typename SDMM>
void copy_sdmm(JMM& jmm, SDMM& sdmm) {
    size_t NComponents = jmm.nComponents();
    enoki::set_slices(sdmm, NComponents);
    copy_means(jmm, sdmm, std::make_index_sequence<SDMM::Embedded::Size>{});
    copy_covs(jmm, sdmm, std::make_index_sequence<SDMM::Tangent::Size * SDMM::Tangent::Size>{});
    for(size_t component_i = 0; component_i < NComponents; ++component_i) {
        enoki::slice(sdmm.weight.pmf, component_i) = jmm.weights()[component_i];
    }
    bool prepare_success = sdmm::prepare(sdmm);
    CHECK_EQ(prepare_success, true);
}

template<typename Value, typename JointSDMM, size_t NComponents>
void init_sdmm(JointSDMM& distribution) {
    enoki::set_slices(distribution, NComponents);
    distribution = enoki::zero<JointSDMM>(NComponents);
    distribution.tangent_space.set_mean(
        sdmm::embedded_t<JointSDMM>{
            enoki::full<Value>(1, NComponents),
            enoki::full<Value>(1, NComponents),
            enoki::full<Value>(1, NComponents),
            enoki::full<Value>(0, NComponents),
            enoki::full<Value>(0, NComponents),
            enoki::full<Value>(1, NComponents)
        }
    );
    distribution.weight.pmf =
        enoki::full<Value>(1.f / NComponents, NComponents);
    spdlog::info("pmf={}", distribution.weight.pmf);
    distribution.cov = enoki::identity<sdmm::matrix_t<JointSDMM>>(NComponents);
    bool prepare_success = sdmm::prepare(distribution);
    CHECK_EQ(prepare_success, true);
}

TEST_CASE("sampling/pdf comparison") {
    constexpr static size_t PacketSize = 8;
    constexpr static size_t JointSize = 5;
    constexpr static size_t MarginalSize = 3;
    constexpr static size_t ConditionalSize = 2;
    constexpr static size_t NComponents = 64;
    constexpr static int NSamples = 1;
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

    using RNG = enoki::PCG32<Value, NSamples>;

    RNG rng;
    RNG jmm_rng;
    enoki::PCG32<float> init_rng;

    #if COMPARISON == 1
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

    auto jmm_distribution = MM();
    auto conditional = MMCond();
    jmm_distribution.setNComponents(t_components);
    conditional.setNComponents(t_components);
    for(int i = 0; i < t_components; ++i) {
        MM::Vectord mean;
        mean << 
            init_rng.next_float32(),
            init_rng.next_float32(),
            init_rng.next_float32(),
            init_rng.next_float32(),
            init_rng.next_float32(),
            init_rng.next_float32();
        mean.bottomRows(3) /= mean.bottomRows(3).norm();
        jmm_distribution.weights()[i] = 1.f / Scalar(t_components);
        jmm_distribution.components()[i].set(
            mean,
            MM::Matrixd::Identity()
        );
    }
    jmm_distribution.configure();

    typename MM::ConditionVectord jmm_point({1, 1, 1});
    Scalar heuristicWeight;
    
    jmm_distribution.conditional(jmm_point, conditional, heuristicWeight);
    auto jmm_sample = conditional.sample(
        [&jmm_rng]() mutable -> Scalar { return jmm_rng.next_float32().coeff(0); }
    );

    Scalar jmm_pdf = conditional.pdf(jmm_sample);
    spdlog::info("jmm_sample={}", jmm_sample.transpose());
    spdlog::info("jmm_pdf={}", jmm_pdf);

    #endif // COMPARISON == 1

    JointSDMM distribution;
    copy_sdmm(jmm_distribution, distribution);

    Conditioner conditioner;
    enoki::set_slices(conditioner, enoki::slices(distribution));
    sdmm::prepare(conditioner, distribution);

    sdmm::embedded_s_t<MarginalSDMM> point({1, 1, 1});
    sdmm::create_conditional(conditioner, point);

    Value inv_jacobian;
    enoki::set_slices(inv_jacobian, enoki::slices(distribution));
    sdmm::replace_embedded_t<ConditionalSDMM, Value> sample;
    sdmm::replace_tangent_t<ConditionalSDMM, Value> tangent_sample;
    
    conditioner.conditional.sample(rng, sample, inv_jacobian, tangent_sample);
    spdlog::info("sample={}", sample);
    spdlog::info("inv_jacobian={}", inv_jacobian);

    Value posterior;
    enoki::set_slices(posterior, enoki::slices(distribution));
    enoki::vectorize_safe(
        VECTORIZE_WRAP_MEMBER(posterior),
        conditioner.conditional,
        sample,
        posterior
    );
    float pdf = enoki::hsum_nested(posterior);
    spdlog::info("pdf={}", pdf);
}
