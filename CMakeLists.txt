cmake_minimum_required(VERSION 3.14)

project(multithreaded-alltoall)

find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG v2.5.0
  EXCLUDE_FROM_ALL SYSTEM)
FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 11.2.0
  EXCLUDE_FROM_ALL SYSTEM)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping
  GIT_TAG feature/session-model
  EXCLUDE_FROM_ALL SYSTEM)

FetchContent_MakeAvailable(CLI11 fmt kamping)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE MPI::MPI_CXX CLI11::CLI11
                                        OpenMP::OpenMP_CXX fmt::fmt kamping::kamping)
target_compile_features(benchmark PRIVATE cxx_std_20)
