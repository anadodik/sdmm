# Copyright (c) 2017 by Ana Dodik, ETH Zurich.
cmake_minimum_required(VERSION 3.13.4)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(violet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(WARNING_FLAGS "-Wall -Wextra -Wfloat-conversion -Wno-deprecated-declarations")
    set(MATH_FLAGS "-funsafe-math-optimizations -fno-math-errno  ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g ${WARNING_FLAGS} ${MATH_FLAGS} -stdlib=libc++ -fPIC -pipe -march=nocona -msse2 -mfpmath=sse  -fomit-frame-pointer -fopenmp -fvisibility=hidden")
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(WARNING_FLAGS "-Wall -Wextra -Wfloat-conversion -Wno-deprecated-declarations")
    set(MATH_FLAGS "-funsafe-math-optimizations -fno-math-errno -fno-signaling-nans -fno-rounding-math ")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g ${WARNING_FLAGS} ${MATH_FLAGS} -fPIC -pipe -march=nocona -msse2 -mfpmath=sse  -fomit-frame-pointer -fopenmp -fvisibility=hidden")
else ()
    MESSAGE(ERROR "Compiler not supported!")
endif()

find_package(OpenMP)

find_package(Boost 1.66 REQUIRED COMPONENTS
    system
    filesystem
    thread
    iostreams
    serialization
    program_options
)

find_package(spdlog CONFIG REQUIRED)
find_package(fmt REQUIRED)

add_executable(violet
    src/main.cpp

    src/figure.cpp
)

target_include_directories(violet PUBLIC
    include
    ext
    ${Boost_INCLUDE_DIRS}
)

target_compile_definitions(violet PRIVATE SPDLOG_FMT_EXTERNAL)
target_link_libraries(violet PUBLIC
    OpenMP::OpenMP_CXX
    fmt::fmt
    spdlog::spdlog
    Boost::system
    Boost::filesystem
    Boost::thread
    Boost::iostreams
    Boost::serialization
    Boost::program_options
)

set(resources_source "${CMAKE_SOURCE_DIR}/resources")
set(resources_destination "${CMAKE_CURRENT_BINARY_DIR}/resources")
add_custom_command(
    TARGET violet
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${resources_source} ${resources_destination}
    MAIN_DEPENDENCY ${resources_source}/shaders/background.frag
    COMMENT "Copying resources folder from \"${resources_source}\" => \"${resources_destination}\"."
)
