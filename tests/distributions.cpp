#include <doctest/doctest.h>

#include <enoki/array.h>
#include <enoki/matrix.h>
#include <enoki/dynamic.h>

#include "sdmm/distributions/sdmm.h"

#include "utils.h"

TEST_CASE("sdmm::pdf<float>") {
    using Value = float;
    using TangentSpace = 
        sdmm::EuclidianTangentSpace<
            sdmm::Vector<Value, 2>,
            sdmm::Vector<Value, 2>
        >;
    using SDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, TangentSpace
    >;

    SDMM distribution;
    distribution.tangent_space.set_mean(sdmm::vector_t<SDMM>(0));
    distribution.cov = enoki::diag<sdmm::matrix_t<SDMM>>({1, 2});;
    distribution.prepare();

    sdmm::vector_s_t<SDMM> point({1, 2});
    Value pdf(0);
    distribution.pdf_gaussian(point, pdf);

    // Compare to results from NumPy
    Value expected_pdf = 0.025110965476047437f;
    CHECK(approx_equals(pdf, expected_pdf));
}

TEST_CASE("sdmm::pdf<Array>") {
    using Value = enoki::Array<float, 2>;
    using TangentSpace = 
        sdmm::EuclidianTangentSpace<
            sdmm::Vector<Value, 2>,
            sdmm::Vector<Value, 2>
        >;
    using SDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, TangentSpace
    >;
    SDMM distribution;
    distribution.tangent_space.set_mean(sdmm::vector_t<SDMM>(0, 1));
    distribution.cov = sdmm::matrix_t<SDMM>(
        Value(3, 2), Value(0.5, 0),
        Value(0.5, 0), Value(1.4, 0.1)
    );
    distribution.prepare();

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

TEST_CASE("sdmm::pdf<DynamicArray>") {
    using Packet = enoki::Array<float, 2>;
    using Value = enoki::DynamicArray<Packet>;
    using TangentSpace = 
        sdmm::EuclidianTangentSpace<
            sdmm::Vector<Value, 2>,
            sdmm::Vector<Value, 2>
        >;
    using SDMM = sdmm::SDMM<
        sdmm::Vector<Value, 2>, sdmm::Matrix<Value, 2>, TangentSpace
    >;
    SDMM distribution;
    enoki::set_slices(distribution, 2);
    distribution.tangent_space.set_mean(
        sdmm::vector_t<SDMM>(
            Value(0, 0), Value(1, 1)
        )
    );
    distribution.cov = sdmm::matrix_t<SDMM>(
        Value(3, 2), Value(0.5, 0),
        Value(0.5, 0), Value(1.4, 0.1)
    );
    enoki::vectorize(
        VECTORIZE_WRAP_MEMBER(prepare),
        distribution
    );

    sdmm::EuclidianTangentSpace<sdmm::Vector<Value, 3>, sdmm::Vector<Value, 3>> tangent_space;

    Value pdf(0);
    enoki::set_slices(pdf, 2);
    SUBCASE("Calculating pdf for point={1, 2}.") {
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
        sdmm::vector_s_t<SDMM> point({0, 0});
        distribution.pdf_gaussian(point, pdf);
        Value expected_pdf({
            0.054777174730721315f,
            0.002397909146739601f
        });
        CHECK(approx_equals(pdf, expected_pdf));
    }

    SUBCASE("Calculating pdf for point={0, 0}, external tangent vector.") {
        sdmm::vector_s_t<SDMM> point({0, 0});
        sdmm::vector_t<SDMM> tangent_vectors;
        distribution.pdf_gaussian(point, pdf, tangent_vectors);
        CHECK(approx_equals(tangent_vectors, sdmm::vector_t<SDMM>(
            Value(0, 0), Value(-1, -1)
        )));
    }
}
