#include <doctest/doctest.h>

#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/dynamic.h>

#include "sdmm/distributions/categorical.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"
#include "sdmm/opt/em.h"

#include "utils.h"

#include "jmm/mixture_model.h"
#include "jmm/opt/stepwise_tangent.h"

#define INCLUDE_TRAINING

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
    bool prepare_success = sdmm::prepare_vectorized(sdmm);
    CHECK_EQ(prepare_success, true);
}

template<typename JMMData, typename SDMMData, typename RNG>
void create_data(JMMData& jmm_data, SDMMData& sdmm_data, RNG& rng, size_t n_points) {
    for(size_t point_i = 0; point_i < n_points; ++point_i) {
        typename JMMData::Vectord point;
        point << 
            rng.next_float32(), rng.next_float32(), rng.next_float32(),
            rng.next_float32(), rng.next_float32(), rng.next_float32();
        point.bottomRows(3) /= point.bottomRows(3).norm();
        Scalar weight = rng.next_float32();
        bool isDiffuse = rng.next_float32() > 0.5;
        Scalar heuristic_pdf = isDiffuse ? rng.next_float32() : 0;

        jmm_data.samples.col(point_i) << point;
        jmm_data.weights(point_i) = weight;
        jmm_data.isDiffuse(point_i) = isDiffuse;
        jmm_data.heuristicPdfs(point_i) = heuristic_pdf;

        sdmm_data.push_back(
            sdmm::embedded_s_t<SDMMData>{
                point(0), point(1), point(2), point(3), point(4), point(5)
            },
            sdmm::normal_s_t<SDMMData>{0, 0, 1},
            weight,
            heuristic_pdf
        );
    }
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
    bool prepare_success = sdmm::prepare_vectorized(distribution);
    CHECK_EQ(prepare_success, true);
}

TEST_CASE("sampling/pdf comparison") {
    constexpr static size_t PacketSize = 8;
    constexpr static size_t JointSize = 5;
    constexpr static size_t MarginalSize = 3;
    constexpr static size_t ConditionalSize = 2;
    constexpr static size_t NComponents = 16;
    constexpr static int NPoints = 20;
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

    using EM = sdmm::EM<JointSDMM>;
    using Data = sdmm::Data<JointSDMM>;
    using JointCov = sdmm::matrix_t<JointSDMM>;

    using RNG = enoki::PCG32<float>;
    RNG rng;
    RNG jmm_rng;
    RNG init_rng;

    constexpr static int t_dims = 6;
    constexpr static int t_conditionalDims = 3;
    constexpr static int t_conditionDims = t_dims - t_conditionalDims;
    constexpr static int t_initComponents = 16;
    constexpr static int t_components = 16;
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
    using StepwiseTangentEM = jmm::StepwiseTangentEM<
        t_dims,
        t_components,
        t_conditionalDims,
        Scalar,
        jmm::MultivariateTangentNormal,
        jmm::MultivariateNormal
    >;
    using Samples = jmm::Samples<t_dims, Scalar>;

    for(int ctr = 0; ctr < 1; ++ctr) {
        auto jmm_distribution = MM();
        auto jmm_conditional = MMCond();
        jmm_distribution.setNComponents(t_components);
        jmm_conditional.setNComponents(t_components);
        StepwiseTangentEM optimizer(
            0.5,
            Eigen::Matrix<Scalar, 5, 1>::Identity() * 1e-5,
            1.f / NComponents
        );
        for(int i = 0; i < t_components; ++i) {
            MM::Matrixd cov_prior; cov_prior <<
                1e-4, 0, 0, 0, 0,
                0, 1e-4, 0, 0, 0,
                0, 0, 1e-4, 0, 0,
                0, 0, 0, 1e-5, 0,
                0, 0, 0, 0, 1e-5;
            MM::Matrixd cov = MM::Matrixd::Random();
            cov = cov.transpose() * cov;
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
                cov
            );
            optimizer.getBPriors()[i] = cov_prior;
        }
        jmm_distribution.configure();

        Samples samples;
        samples.reserve(NPoints);
        samples.setSize(NPoints);

        JointSDMM distribution;
        copy_sdmm(jmm_distribution, distribution);

        // std::cerr << "jmm_cov_sqrt=[";
        // for(auto& component : jmm_distribution.components()) {
        //     std::cerr << component.cholL() << ",\n";
        // }
        // std::cerr << "]\n";
        // spdlog::info("cov_sqrt={}", distribution.cov_sqrt);

        EM em;
        em = enoki::zero<EM>(NComponents);
        Scalar weight_prior = 1.f / NComponents;
        Scalar cov_prior_strength = 5.f / NComponents;
        JointCov cov_prior = sdmm::full_inner<JointCov, Value, NComponents>(
            2e-3, 0, 0, 0, 0,
            0, 2e-3, 0, 0, 0,
            0, 0, 2e-3, 0, 0,
            0, 0, 0, 2e-4, 0,
            0, 0, 0, 0, 2e-4
        );
        em.set_priors(weight_prior, cov_prior_strength, cov_prior);

        Data data;
        data.reserve(NPoints + 200);

        create_data(samples, data, init_rng, NPoints);

        #ifdef INCLUDE_TRAINING
        size_t iterations = 4;
        Scalar max_error;
        for(size_t it = 0; it < iterations; ++it) {
            optimizer.optimize(jmm_distribution, samples, max_error);
            // std::cerr << "jmm_weights(" << it << ")=[";
            // for(auto& weight : jmm_distribution.weights()) {
            //     std::cerr << weight << ", ";
            // }
            // std::cerr << "]\n";

            // std::cerr << "jmm_means(" << it << ")=[";
            // for(auto& component : jmm_distribution.components()) {
            //     std::cerr << component.mean() << "\n";
            // }
            // std::cerr << "]\n";

            // std::cerr << "jmm_cov_sqrts(" << it << ")=[";
            // for(auto& component : jmm_distribution.components()) {
            //     std::cerr << component.det() << "\n";
            // }
            // std::cerr << "]\n";

            std::cerr << "jmm_stats=[";
            for(auto& weight : optimizer.getStatsGlobal().weights) {
                std::cerr << weight << ", ";
            }
            std::cerr << "]\n";
        }


        // TODO: emIt
        for(size_t it = 0; it < iterations; ++it) {
            em.step(distribution, data);
            // spdlog::info("weights({})={}", it, distribution.weight.pmf);
            // spdlog::info("means({})={}", it, distribution.tangent_space.mean);
            // spdlog::info("cov_sqrts({})={}", it, distribution.inv_cov_sqrt_det);
            spdlog::info("stats={}", em.stats.weight);
        }
        // spdlog::info("weights={}", distribution.weight.pmf);
        #endif // INCLUDE_TRAINING

        typename MM::ConditionVectord jmm_point({
            jmm_rng.next_float32(),
            jmm_rng.next_float32(),
            jmm_rng.next_float32()
        });
        Scalar heuristicWeight;
        
        // jmm_distribution.conditional(jmm_point, jmm_conditional, heuristicWeight);
        // auto jmm_sample = jmm_conditional.sample(
        //     [&jmm_rng]() mutable -> Scalar { return jmm_rng.next_float32(); }
        // );

        // Scalar jmm_pdf = jmm_conditional.pdf(jmm_sample);
        // spdlog::info("jmm_sample={}", jmm_sample.transpose());
        // spdlog::info("jmm_pdf={}", jmm_pdf);

        auto jmm_dist_sample = jmm_distribution.sample(
            [&jmm_rng]() mutable -> Scalar { return jmm_rng.next_float32(); }
        );
        Scalar jmm_dist_pdf = jmm_distribution.pdf(jmm_dist_sample);
        spdlog::info("jmm_dist_sample={}", jmm_dist_sample.transpose());
        spdlog::info("jmm_dist_pdf={}", jmm_dist_pdf);

        Conditioner conditioner;
        enoki::set_slices(conditioner, enoki::slices(distribution));
        sdmm::prepare(conditioner, distribution);

        ConditionalSDMM conditional;
        enoki::set_slices(conditional, enoki::slices(distribution));

        sdmm::embedded_s_t<MarginalSDMM> point({
            rng.next_float32(),
            rng.next_float32(),
            rng.next_float32()
        });
        // sdmm::create_conditional(conditioner, point, conditional);

        Scalar inv_jacobian;
        sdmm::replace_embedded_t<ConditionalSDMM, Scalar> sample;
        sdmm::replace_tangent_t<ConditionalSDMM, Scalar> tangent_sample;
        
        Value posterior;
        Scalar pdf;
        enoki::set_slices(posterior, enoki::slices(distribution));
        // conditional.sample(rng, sample, inv_jacobian, tangent_sample);
        // enoki::vectorize_safe(
        //     VECTORIZE_WRAP_MEMBER(posterior),
        //     conditional,
        //     sample,
        //     posterior
        // );
        // pdf = enoki::hsum_nested(posterior);
        // spdlog::info("sample={}, norm={}", sample, enoki::norm(sample));
        // spdlog::info("pdf={}", pdf);

        sdmm::replace_embedded_t<JointSDMM, Scalar> dist_sample;
        sdmm::replace_tangent_t<JointSDMM, Scalar> dist_tangent_sample;
        distribution.sample(rng, dist_sample, inv_jacobian, dist_tangent_sample);
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(posterior),
            distribution,
            dist_sample,
            posterior
        );
        pdf = enoki::hsum_nested(posterior);
        spdlog::info("dist_sample={}", dist_sample);
        spdlog::info("dist_pdf={}", pdf);
    }
}
