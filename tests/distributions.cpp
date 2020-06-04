#include <doctest/doctest.h>

#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/dynamic.h>

#include "sdmm/distributions/categorical.h"
#include "sdmm/distributions/sdmm.h"
#include "sdmm/distributions/sdmm_conditioner.h"

#include "utils.h"

TEST_CASE("SDMM::pdf<float>") {
    using Value = float;
    using TangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, 2>, sdmm::Vector<Value, 2>
    >;
    using SDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, TangentSpace
    >;

    SDMM distribution;
    distribution.weight.pmf = 1;
    distribution.tangent_space.set_mean(sdmm::vector_t<SDMM>(0));
    distribution.cov = enoki::diag<sdmm::matrix_t<SDMM>>({1, 2});;
    distribution.prepare_cov();

    sdmm::vector_s_t<SDMM> point({1, 2});
    Value pdf(0);
    distribution.pdf_gaussian(point, pdf);

    // Compare to results from NumPy
    Value expected_pdf = 0.025110965476047437f;
    CHECK(approx_equals(pdf, expected_pdf));
}

TEST_CASE("SDMM::pdf<Array>") {
    using Value = enoki::Array<float, 2>;
    using TangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, 2>, sdmm::Vector<Value, 2>
    >;
    using SDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, TangentSpace
    >;
    SDMM distribution;
    distribution.weight.pmf = Value(0.5, 0.5);
    distribution.tangent_space.set_mean(sdmm::vector_t<SDMM>(0, 1));
    distribution.cov = sdmm::matrix_t<SDMM>(
        Value(3, 2), Value(0.5, 0),
        Value(0.5, 0), Value(1.4, 0.1)
    );
    CHECK(sdmm::prepare(distribution));

    Value pdf(0);
    SUBCASE("Calculating pdf for point={1, 2}.") {
        sdmm::vector_s_t<SDMM> point({1, 2});
        distribution.pdf_gaussian(point, pdf);

        // Compare to results from NumPy
        Value expected_pdf({
            0.05207269256276517f,
            0.0018674935212148857f
        });
        CHECK(approx_equals(pdf, expected_pdf));
    }

    SUBCASE("Calculating pdf for point={0, 0}.") {
        sdmm::vector_s_t<SDMM> point({0, 0});
        distribution.pdf_gaussian(point, pdf);
        Value expected_pdf({
            0.054777174730721315f,
            0.002397909146739601f
        });
        CHECK(approx_equals(pdf, expected_pdf));
    }
}

TEST_CASE("SDMM::pdf<DynamicArray>") {
    using Packet = enoki::Packet<float, 2>;
    using Value = enoki::DynamicArray<Packet>;
    using TangentSpace = sdmm::EuclidianTangentSpace<
        sdmm::Vector<Value, 2>, sdmm::Vector<Value, 2>
    >;
    using SDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, TangentSpace
    >;
    SDMM distribution;
    enoki::set_slices(distribution, 2);
    distribution.weight.pmf = Value(0.2, 0.8);
    distribution.tangent_space.set_mean(
        sdmm::vector_t<SDMM>(
            Value(0, 0), Value(1, 1)
        )
    );
    distribution.cov = sdmm::matrix_t<SDMM>(
        Value(3, 2), Value(0.5, 0),
        Value(0.5, 0), Value(1.4, 0.1)
    );

    CHECK(sdmm::prepare(distribution));

    sdmm::EuclidianTangentSpace<sdmm::Vector<Value, 3>, sdmm::Vector<Value, 3>> tangent_space;

    SUBCASE("Calculating pdf for point={1, 2}.") {
        Value pdf(0);
        enoki::set_slices(pdf, 2);
        sdmm::vector_s_t<SDMM> point({1, 2});
        enoki::vectorize(
            VECTORIZE_WRAP_MEMBER(pdf_gaussian),
            distribution,
            point,
            pdf
        );

        // Compare to results from NumPy
        Value expected_pdf({
            0.05207269256276517f,
            0.0018674935212148857f
        });
        CHECK(approx_equals(pdf, expected_pdf));
    }

    SUBCASE("Calculating pdf for point={0, 0}.") {
        Value pdf(0);
        enoki::set_slices(pdf, 2);
        sdmm::vector_s_t<SDMM> point({0, 0});
        enoki::vectorize(
            VECTORIZE_WRAP_MEMBER(pdf_gaussian),
            distribution,
            point,
            pdf
        );
        Value expected_pdf({
            0.054777174730721315f,
            0.002397909146739601f
        });
        CHECK(approx_equals(pdf, expected_pdf));
    }

    SUBCASE("Calculating pdf for point={0, 0}, external tangent vector.") {
        Value pdf(0);
        enoki::set_slices(pdf, 2);
        sdmm::vector_t<SDMM> tangent_vectors;
        enoki::set_slices(tangent_vectors, 2);
        sdmm::vector_s_t<SDMM> point({0, 0});
        enoki::vectorize(
            VECTORIZE_WRAP_MEMBER(pdf_gaussian),
            distribution,
            point,
            pdf,
            tangent_vectors
        );
        CHECK(approx_equals(tangent_vectors, sdmm::vector_t<SDMM>(
            Value(0, 0), Value(-1, -1)
        )));
    }

    SUBCASE("Calculating posterior for point={1, 2}.") {
        Value posterior(0);
        enoki::set_slices(posterior, 2);
        sdmm::vector_t<SDMM> tangent_vectors;
        enoki::set_slices(tangent_vectors, 2);
        sdmm::vector_s_t<SDMM> point({1, 2});
        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(posterior),
            distribution,
            point,
            posterior,
            tangent_vectors
        );

        // Compare to results from NumPy
        Value expected_posterior({
            0.2f * 0.05207269256276517f,
            0.8f * 0.0018674935212148857f
        });
        CHECK(approx_equals(posterior, expected_posterior));
    }


    SUBCASE("Conditioner") {
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
        enoki::set_slices(distribution, 2);
        distribution.weight.pmf = Value(0.2, 0.8);
        distribution.tangent_space.set_mean(
            sdmm::vector_t<JointSDMM>(
                Value(0, 0), Value(1, 1), Value(2, 2), Value(-1, 1)
            )
        );
        distribution.cov = sdmm::matrix_t<JointSDMM>(
            Value(3, 2), Value(0.5, 0), Value(0.1, 0.1), Value(0.4, 0.4),
            Value(0.5, 0), Value(1.4, 0.5), Value(0, 0), Value(0.2, 0.2),
            Value(0.1, 0.1), Value(0, 0), Value(1, 1), Value(0.2, 0.2),
            Value(0.4, 0.4), Value(0.2, 0.2), Value(0.2, 0.2), Value(3, 3)
        );

        CHECK(sdmm::prepare(distribution));
        Conditioner conditioner;
        enoki::set_slices(conditioner, enoki::slices(distribution));
        
        enoki::vectorize(
            VECTORIZE_WRAP(create_marginal),
            distribution,
            conditioner.marginal
        );

        enoki::vectorize_safe(
            VECTORIZE_WRAP_MEMBER(prepare_vectorized),
            conditioner,
            distribution
        );

        typename ConditionalSDMM::Matrix expected_cov{
            Value(0.99645569620253160447, 0.99499999999999999556),
            Value(0.18835443037974683444, 0.17999999999999999333),
            Value(0.18835443037974683444, 0.17999999999999999333),
            Value(2.93316455696202549319, 2.83999999999999985789)
        };

        CHECK(enoki::slice(expected_cov, 0) == enoki::slice(conditioner.conditional.cov, 0));
        CHECK(enoki::slice(expected_cov, 1) == enoki::slice(conditioner.conditional.cov, 1));

        sdmm::vector_s_t<MarginalSDMM> point({1, 2});
        enoki::vectorize(
            VECTORIZE_WRAP_MEMBER(create_conditional_vectorized),
            conditioner,
            point
        );
        CHECK(conditioner.conditional.weight.prepare());

        sdmm::prepare(conditioner, distribution);
        sdmm::create_conditional(conditioner, point);

        CHECK(enoki::slice(expected_cov, 0) == enoki::slice(conditioner.conditional.cov, 0));
        CHECK(enoki::slice(expected_cov, 1) == enoki::slice(conditioner.conditional.cov, 1));

        typename ConditionalSDMM::Vector expected_mean{
            Value(2.02278481012658, 2.05000000000000),
            Value(-0.78227848101265, 1.60000000000000)
        };
        CHECK(enoki::slice(expected_mean, 0) == enoki::slice(conditioner.conditional.tangent_space.mean, 0));
        CHECK(enoki::slice(expected_mean, 1) == enoki::slice(conditioner.conditional.tangent_space.mean, 1));
    }
}

TEST_CASE("Categorical<Value>") {
    using Packet = enoki::Packet<float, 2>;
    using DynamicValue = enoki::DynamicArray<Packet>;
    using Color = sdmm::Vector<DynamicValue, 3>;
    using Categorical = sdmm::Categorical<Color>;
    using BoolOuter = typename Categorical::BoolOuter;

    SUBCASE("Categorical::is_valid()") {
        Categorical categorical = enoki::zero<Categorical>(5);
        CHECK(categorical.is_valid() == BoolOuter{false, false, false});
        categorical.pmf.x() = enoki::full<DynamicValue>(1, 5);
        categorical.pmf.y() = enoki::full<DynamicValue>(1, 5);
        CHECK(categorical.is_valid() == BoolOuter{true, true, false});
        categorical.pmf.z() = enoki::full<DynamicValue>(1, 5);
        CHECK(categorical.is_valid() == BoolOuter{true, true, true});
        categorical.pmf.z().coeff(0) = 0.f;
        CHECK(categorical.is_valid() == BoolOuter{true, true, true});
    }

    SUBCASE("Categorical::prepare()") {
        Categorical categorical = enoki::zero<Categorical>(5);
        enoki::set_slices(categorical, 5);
        categorical.pmf = enoki::full<Color>(1, 5);
        categorical.pmf.z() = enoki::zero<DynamicValue>(5);
        CHECK(categorical.prepare() == BoolOuter{true, true, false});
        Color expected = (enoki::arange<DynamicValue>(5) + 1) / 5.f;
        expected.z() = enoki::zero<DynamicValue>(5);
        CHECK(categorical.cdf == expected);
    }
}

