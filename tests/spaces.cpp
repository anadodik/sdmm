#include <doctest/doctest.h>

#include <enoki/dynamic.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/utils.h"
#include "sdmm/spaces/directional.h"
#include "sdmm/spaces/spatio_directional.h"
#include "sdmm/distributions/sdmm.h"

#include "utils.h"

TEST_CASE("DirectionalTangentSpace") {
    using Packet = enoki::Packet<float>;
    using Value = enoki::DynamicArray<Packet>;
    using Embedded = sdmm::Vector<Value, 3>; 
    using Tangent = sdmm::Vector<Value, 2>; 
    using TangentSpace = sdmm::DirectionalTangentSpace<Embedded, Tangent>;

    TangentSpace tangent_space = enoki::zero<TangentSpace>(2);
    tangent_space.set_mean(Embedded{0, 1, 0});
    Value inv_jacobian;
    enoki::set_slices(inv_jacobian, 2);
    spdlog::info(tangent_space.to(Embedded{0, 0, 1}, inv_jacobian));
    spdlog::info(tangent_space.to(Embedded{0, 1, 0}, inv_jacobian));

    spdlog::info("ts.from={}", tangent_space.coordinate_system.from * Embedded{0, 0, 1});
}

TEST_CASE("SpatioDirectionalTangentSpace") {
    using Packet = enoki::Packet<float>;
    using Value = enoki::DynamicArray<Packet>;
    using Embedded = sdmm::Vector<Value, 4>; 
    using Tangent = sdmm::Vector<Value, 3>; 
    using TangentSpace = sdmm::SpatioDirectionalTangentSpace<Embedded, Tangent>;

    TangentSpace tangent_space = enoki::zero<TangentSpace>(2);
    tangent_space.set_mean(Embedded{1, 0, 1, 0});
    Value inv_jacobian;
    enoki::set_slices(inv_jacobian, 2);
    spdlog::info(tangent_space.to(Embedded{1, 0, 0, 1}, inv_jacobian));
    spdlog::info(tangent_space.to(Embedded{0, 0, 1, 0}, inv_jacobian));

    // spdlog::info("ts.from={}", tangent_space.coordinate_system.from * Embedded{1, 0, 0, 1});
}
