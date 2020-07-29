# Copyright (c) 2020 by Ana Dodik
cmake_minimum_required(VERSION 3.13.4)

# Benchmarks executable.
add_executable(
    benchmarks

    distributions.cpp
    em.cpp
    sampling.cpp
)

target_compile_definitions(benchmarks PRIVATE COMPARISON=0)

target_compile_features(benchmarks PRIVATE cxx_std_17)

target_include_directories(benchmarks PRIVATE benchmark::benchmark)

target_link_libraries(benchmarks PRIVATE sdmm ${CONAN_LIBS} jmm jmm_interface)
