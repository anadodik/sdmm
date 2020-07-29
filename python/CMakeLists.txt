# Copyright (c) 2020 by Ana Dodik
cmake_minimum_required(VERSION 3.13.4)

project(pysdmm LANGUAGES CXX)

pybind11_add_module(
    pysdmm

    bindings.cpp
)

target_compile_features(pysdmm PRIVATE cxx_std_17)

target_link_libraries(pysdmm PRIVATE sdmm ${CONAN_LIBS})

