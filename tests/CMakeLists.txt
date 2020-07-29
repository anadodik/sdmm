# Copyright (c) 2020 by Ana Dodik
cmake_minimum_required(VERSION 3.13.4)

# Test executable.
add_executable(
    tests

    tests.cpp
    linalg.cpp
    distributions.cpp
    matrix.cpp
    spaces.cpp
    comparison.cpp
)

target_compile_definitions(tests PRIVATE COMPARISON=1)

target_compile_features(tests PRIVATE cxx_std_17)

target_link_libraries(tests PRIVATE sdmm jmm jmm_interface ${CONAN_LIBS})

