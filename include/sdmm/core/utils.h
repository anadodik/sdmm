#pragma once

#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <enoki/array.h>
#include <enoki/dynamic.h>

#define VECTORIZE_WRAP(FUNC_NAME) [](auto&&... params) { FUNC_NAME(params...); }

namespace sdmm {

template<typename Func, typename... Args>
auto vectorize(const Func& func, Args&&... args) {
    std::tuple packet_sizes{enoki::packets(args)...};
    std::tuple slice_sizes{enoki::slices(args)...};
    spdlog::info("{}", packet_sizes);
}

}
