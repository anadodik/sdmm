#include <doctest/doctest.h>

#include <enoki/dynamic.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/utils.h"

#include "utils.h"

TEST_CASE("sdmm::linalg::Matrix") {
    using Packet = enoki::Packet<float, 4>;
    using Value = enoki::DynamicArray<Packet>;
    using Matrix = sdmm::Matrix<Value, 3, 5>;
    using Vector = sdmm::Vector<Value, 5>;
    Matrix mat = enoki::zero<Matrix>(2);

    for(size_t c = 0; c < Matrix::Cols; ++c) {
        for(size_t r = 0; r < Matrix::Rows; ++r) {
            mat(r, c) = enoki::full<Value>(c + 1, 2);
        }
    }
    Vector vec = enoki::arange<Vector>(5) + 1;

    sdmm::Matrix<Value, 3, 3> mmt_expected{
        Value(55), Value(55), Value(55),
        Value(55), Value(55), Value(55),
        Value(55), Value(55), Value(55),
    };
    
    CHECK(approx_equals(mat * sdmm::linalg::transpose(mat), mmt_expected));

    sdmm::Matrix<Value, 5, 5> mtm_expected{
        Value(3), Value(6), Value(9), Value(12), Value(15),
        Value(6), Value(12), Value(18), Value(24), Value(30),
        Value(9), Value(18), Value(27), Value(36), Value(45),
        Value(12), Value(24), Value(36), Value(48), Value(60),
        Value(15), Value(30), Value(45), Value(60), Value(75),
    };
    CHECK(approx_equals(sdmm::linalg::transpose(mat) * mat, mtm_expected));


    sdmm::Vector<Value, 3> mv_expected{
        Value(55), Value(55), Value(55)
    };
    CHECK(approx_equals(mat * vec, mv_expected));
}
