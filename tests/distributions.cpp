#include <doctest/doctest.h>

#include "sdmm/distributions/sdmm.h"

#include "utils.h"

TEST_CASE("Testing sdmm::pdf scalar test.") {
    using Value = float;
    using SDMM = sdmm::SDMM<Value, 2, 2>;
    SDMM distribution;
    distribution.mean = sdmm::vector_t<SDMM>(0);
    distribution.cov = enoki::diag<sdmm::matrix_t<SDMM>>({1, 2});;
    distribution.prepare();

    sdmm::vector_s_t<SDMM> point({1, 2});
    Value pdf(0);
    distribution.pdf_gaussian(point, pdf);

    // Compare to results from NumPy
    Value expected_pdf = 0.025110965476047437f;
    CHECK(enoki_approx_equals(pdf, expected_pdf));
}

TEST_CASE("Testing sdmm::pdf vector test.") {
    using Value = enoki::Array<float, 2>;
    using SDMM = sdmm::SDMM<Value, 2, 2>;
    SDMM distribution;
    distribution.mean = sdmm::vector_t<SDMM>(0, 1);
    distribution.cov = sdmm::matrix_t<SDMM>(
        Value(3, 2), Value(0.5, 0),
        Value(0.5, 0), Value(1.4, 0.1)
    );
    distribution.prepare();

    Value pdf(0);
    SUBCASE("point={1, 2}.") {
        sdmm::vector_s_t<SDMM> point({1, 2});
        distribution.pdf_gaussian(point, pdf);

        // Compare to results from NumPy
        Value expected_pdf({
            0.05207269256276517f,
            0.0018674935212148857f
        });
        CHECK(enoki_approx_equals(pdf, expected_pdf));
    }

    SUBCASE("point={0, 0}.") {
        sdmm::vector_s_t<SDMM> point({0, 0});
        distribution.pdf_gaussian(point, pdf);
        Value expected_pdf({
            0.054777174730721315f,
            0.002397909146739601f
        });
        CHECK(enoki_approx_equals(pdf, expected_pdf));
    }
}
