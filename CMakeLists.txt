# Copyright (c) 2020 by Ana Dodik
cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(sdmm LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(WARNING_FLAGS "-Wall -Wextra -Wfloat-conversion -Wno-deprecated-declarations")
    set(MATH_FLAGS "-funsafe-math-optimizations -fno-math-errno  ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g ${WARNING_FLAGS} ${MATH_FLAGS} -stdlib=libc++ -march=native")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(WARNING_FLAGS "-ftrack-macro-expansion=0 -Wall -Wextra -Wfloat-conversion -Wno-deprecated-declarations -Wno-misleading-indentation")
    set(MATH_FLAGS "-funsafe-math-optimizations -fno-math-errno -fno-signaling-nans -fno-rounding-math")
    set(SANITIZER_FLAGS "")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -g ${WARNING_FLAGS} ${MATH_FLAGS} ${SANITIZER_FLAGS}")
else ()
    MESSAGE(ERROR "Compiler not supported!")
endif()

# Adding conan dependencies.
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Adding enoki dependency.
add_subdirectory(ext/enoki)
enoki_set_compile_flags()
enoki_set_native_flags()

# Adding JMM dependency for comparisons.
add_library(jmm ext/jmm/include/jmm/multivariate_tangent_normal.cpp)
target_include_directories(jmm PRIVATE ext/jmm/include)
target_link_libraries(jmm PUBLIC ${CONAN_LIBS})
target_compile_options(jmm PRIVATE 
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-w>
    $<$<CXX_COMPILER_ID:MSVC>:/W1>
)
add_library(jmm_interface INTERFACE)
target_include_directories(jmm_interface INTERFACE ext/jmm/include)
target_compile_options(jmm_interface INTERFACE 
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-w>
    $<$<CXX_COMPILER_ID:MSVC>:/W1>
)

# The main SDMM library.
add_library(sdmm INTERFACE)
target_include_directories(sdmm INTERFACE include ext/enoki/include)
target_link_libraries(sdmm INTERFACE ${CONAN_LIBS})
target_compile_features(sdmm INTERFACE cxx_std_17)

add_subdirectory(tests)

add_subdirectory(benchmarks)

add_subdirectory(python)
