#include <doctest/doctest.h>

#include <Eigen/Cholesky>

#include "sdmm/linalg/cholesky.h"

#include "utils.h"

TEST_CASE("Testing sdmm::linalg::choleky.") {
    static constexpr int MatSize = 3;
    static constexpr int ArraySize = 2;
    using Matrix3f = Eigen::Matrix<double, MatSize, MatSize>;

    using Array = enoki::Array<float, ArraySize>;
    using Matrix = enoki::Matrix<Array, MatSize>;
    using Mask = enoki::mask_t<Array>;
        
    std::array<Matrix3f, ArraySize> eigen_mats;
    eigen_mats[0] <<
        0.0352478, 0.294885, 0.13223, 
        0.294885, 2.46774, 1.10656, 
        0.13223, 1.10656, 0.496206;
    eigen_mats[1] <<
        0.0752945, 0.203741, 0.0132939,
        0.203741, 0.551391, 0.0359771,
        0.0132939, 0.0359771, 0.00235748;

    Matrix enoki_mat;
    for(int r = 0; r < MatSize; ++r) {
        for(int c = 0; c < MatSize; ++c) {
            enoki_mat(r, c) = Array{eigen_mats[0](r, c), eigen_mats[1](r, c)};
        }
    }
    Matrix enoki_result;
    Mask is_psd;
    sdmm::linalg::cholesky(enoki_mat, enoki_result, is_psd);
    
    SUBCASE("Checking A = L * L^T for consistency.") {
        CHECK(approx_equals(
            enoki_result * enoki::transpose(enoki_result), enoki_mat
        ));
    }

    SUBCASE("Checking against Eigen::LLT.") {
        for(int mat_i = 0; mat_i < ArraySize; ++mat_i) {
            Eigen::LLT<Matrix3f> llt(eigen_mats[mat_i]);
            Matrix3f eigen_result = llt.matrixL();
            approx_equals_lower_tri(
                enoki_result,
                eigen_result
            );
        }
    }

    SUBCASE("Checking solve for consistency.") {
        using SingleVector = enoki::Array<float, MatSize>;
        using Vector = enoki::Array<Array, MatSize>;

        SingleVector b({1, 2, 0.5});
        Vector x;
        sdmm::linalg::solve(enoki_result, b, x);
        Vector b_check = enoki_result * x;
        CHECK(approx_equals(b, b_check));
    }
}

