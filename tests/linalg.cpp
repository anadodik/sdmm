#include <doctest/doctest.h>

#include <Eigen/Cholesky>

#include <enoki/dynamic.h>

#include "sdmm/core/utils.h"
#include "sdmm/linalg/cholesky.h"

#include "utils.h"

template<typename Value, typename Enable=void>
struct packet_size;

template<typename Value>
struct packet_size<Value, std::enable_if_t<enoki::is_dynamic_v<Value>>> {
    static constexpr size_t value = Value::PacketSize;
};

template<typename Value>
struct packet_size<Value, std::enable_if_t<!enoki::is_dynamic_v<Value>>> {
    static constexpr size_t value = Value::Size;
};

template<typename Value>
static constexpr size_t packet_size_v = packet_size<Value>::value;


template<typename Value>
void test_cholesky() {
    static constexpr size_t ArraySize = packet_size_v<Value>;
    static constexpr size_t MatSize = 3;
    using Matrix3f = Eigen::Matrix<float, MatSize, MatSize>;

    using Vector = enoki::Array<Value, MatSize>;
    using Matrix = enoki::Matrix<Value, MatSize>;
    using Mask = enoki::mask_t<Value>;

    std::array<Matrix3f, ArraySize> eigen_mats;
    eigen_mats[0] <<
        0.0352478f, 0.294885f, 0.13223f, 
        0.294885f, 2.46774f, 1.10656f, 
        0.13223f, 1.10656f, 0.496206f;
    eigen_mats[1] <<
        0.0752945f, 0.203741f, 0.0132939f,
        0.203741f, 0.551391f, 0.0359771f,
        0.0132939f, 0.0359771f, 0.00235748f;

    Matrix enoki_mat;
    set_slices(enoki_mat, 2);
    for(size_t r = 0; r < MatSize; ++r) {
        for(size_t c = 0; c < MatSize; ++c) {
            enoki_mat(r, c) = Value{eigen_mats[0](r, c), eigen_mats[1](r, c)};
        }
    }

    Matrix enoki_result = enoki::zero<Matrix>(); set_slices(enoki_result, 2);
    Mask is_psd; set_slices(is_psd, 2);
    if constexpr(enoki::is_dynamic_v<Value>) {
        enoki::vectorize(
            VECTORIZE_WRAP(sdmm::linalg::cholesky),
            enoki_mat,
            enoki_result,
            is_psd
        );
    } else {
        sdmm::linalg::cholesky(enoki_mat, enoki_result, is_psd);
    }

    
    SUBCASE("Checking A = L * L^T for consistency.") {
        CHECK(approx_equals(
            enoki_result * enoki::transpose(enoki_result), enoki_mat
        ));
    }

    SUBCASE("Checking against Eigen::LLT.") {
        for(size_t mat_i = 0; mat_i < ArraySize; ++mat_i) {
            Eigen::LLT<Matrix3f> llt(eigen_mats[mat_i]);
            Matrix3f eigen_result = llt.matrixL();
            approx_equals_lower_tri(
                enoki_result,
                eigen_result
            );
        }
    }

    SUBCASE("sdmm::linalg::solve -- consistency check, RHS==scalar") {
        enoki::Array<float, 3> b{1, 2, 0.5};
        Vector x;
        enoki::set_slices(x, 2);
        if constexpr(enoki::is_dynamic_v<Value>) {
            enoki::vectorize_safe(
                VECTORIZE_WRAP(sdmm::linalg::solve),
                enoki_result,
                b,
                x
            );
        } else {
            sdmm::linalg::solve(enoki_result, b, x);
        }
        Vector b_check = enoki_result * x;
        CHECK(approx_equals(b, b_check));
    }

    SUBCASE("sdmm::linalg::solve -- consistency check, RHS==DynArray") {
        Vector b{1, 2, 0.5};
        enoki::set_slices(b, 2);
        b.x() = Value{1, 0};
        b.y() = Value{0, 2};
        b.z() = Value{0, 0.5};
        Vector x;
        enoki::set_slices(x, 2);
        if constexpr(enoki::is_dynamic_v<Value>) {
            enoki::vectorize_safe(
                VECTORIZE_WRAP(sdmm::linalg::solve),
                enoki_result,
                b,
                x
            );
        } else {
            sdmm::linalg::solve(enoki_result, b, x);
        }
        Vector b_check = enoki_result * x;
        CHECK(approx_equals(b, b_check));
    }
}

TEST_CASE("sdmm::linalg::choleky<Packet>") {
    static constexpr size_t ArraySize = 2;
    using Packet = enoki::Packet<float, ArraySize>;

    test_cholesky<Packet>();
}

TEST_CASE("sdmm::linalg::choleky<DynamicArray>") {
    static constexpr size_t ArraySize = 2;
    using Packet = enoki::Packet<float, ArraySize>;
    using DynamicArray = enoki::DynamicArray<Packet>;

    test_cholesky<DynamicArray>();
}


