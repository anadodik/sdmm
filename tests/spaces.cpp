#include <doctest/doctest.h>

#include <enoki/dynamic.h>

#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include "sdmm/core/utils.h"
#include "sdmm/spaces/directional.h"

#include "utils.h"

TEST_CASE("DirectionalTangentSpace") {
    using Packet = enoki::Packet<float>;
    using Value = enoki::DynamicArray<Packet>;
    using Embedded = sdmm::Vector<Value, 3>; 
    using Tangent = sdmm::Vector<Value, 2>; 
    using TangentSpace = sdmm::DirectionalTangentSpace<Embedded, Tangent>;

    TangentSpace tangent_space = enoki::zero<TangentSpace>(2);
    tangent_space.set_mean(Embedded{0, 1, 0});
    spdlog::info(tangent_space.to(Embedded{0, 0, 1}));
    spdlog::info(tangent_space.to(Embedded{0, 1, 0}));

    spdlog::info("ts.from={}", tangent_space.coordinate_system.from * Embedded{0, 0, 1});
}
