cmake_minimum_required(VERSION 3.20)

project(TinyInfer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(BUILD_TESTING "Build tests" ON)
option(TINYINFER_BUILD_PYTHON "Build python binding" OFF)
option(TINYINFER_BUILD_SERVING "Build serving module" OFF)

add_subdirectory(src)
add_subdirectory(examples)

if(TINYINFER_BUILD_SERVING)
  add_subdirectory(serving)
endif()

if(TINYINFER_BUILD_PYTHON)
  include(FetchContent)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  FetchContent_Declare(pybind11 URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip)
  FetchContent_MakeAvailable(pybind11)
  add_subdirectory(python)
endif()

if(BUILD_TESTING)
  include(FetchContent)
  include(CTest)
  enable_testing()

  FetchContent_Declare(googletest URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(tests)

  FetchContent_Declare(googlebenchmark URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googlebenchmark)
  add_subdirectory(benchmarks)
endif()
