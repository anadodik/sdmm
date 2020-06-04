#include <cassert>

#include <enoki/array.h>
#include <enoki/dynamic.h>
#include <enoki/matrix.h>
#include <enoki/random.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/constants.h"
#include "sdmm/linalg/cholesky.h"

#include "sdmm/distributions/sdmm.h"

template<typename Value, size_t Size>
enoki::Array<Value, Size> row(const enoki::Matrix<Value, Size>& matrix, int row) {
    using Array = enoki::Array<Value, Size>;
    using Index = enoki::Array<uint32_t, Size>;

    Index index = enoki::arange<Index>() * Size;
    void* mem = (void*) (matrix.coeff(0).data() + row);
    Array result;
    for (size_t i = 0; i < Array::Size; ++i)
        result[i] = ((Value *) mem)[index[i]];
    return result;
}

template<typename SDMM>
void random_init(const SDMM& distribution) {
    using RNG = enoki::PCG32<typename SDMM::Packet>;
    RNG rng(
        PCG32_DEFAULT_STATE,
        enoki::arange<typename SDMM::Packet>(slices(distribution))
    );
    spdlog::info("rng generated {}.", rng.next_float32());
    spdlog::info("rng generated {}.", rng.next_float32());
    // for (size_t i = 0; i < packets(coord1); ++i) {
    //     packet(distribution, i) = 
    // }
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    // enoki::set_flush_denormals(true);
    spdlog::info("Welcome to SDMM! Using max packet size: {}", enoki::max_packet_size);

    // using Packet = enoki::Packet<float, 32>;
    // using Array = enoki::DynamicArray<Packet>;
    // using SDMM = sdmm::SDMM<Array, 2, 2>;

    // SDMM distribution;
    // enoki::set_slices(distribution, 100);
    // random_init(distribution);
    // distribution.mean = sdmm::vector_t<SDMM>(0, 1);
    // distribution.cov = sdmm::matrix_t<SDMM>(
    //     Value(3, 2), Value(0.5, 0),
    //     Value(0.5, 0), Value(1.4, 0.1)
    // );
    // distribution.prepare();

    // sdmm::vector_s_t<SDMM> point({0, 0});
    // Value pdf(0);
    // distribution.pdf_gaussian(point, pdf);
    // spdlog::info("pdf={}", pdf);
    return 0;
}
