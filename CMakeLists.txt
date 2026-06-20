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
  GIT_TAG main
  EXCLUDE_FROM_ALL SYSTEM)
# kamping v2 + dSTL. We only need the dstl component, so fetch the kamping-v2
# monorepo and use its dstl/ subdirectory as the add_subdirectory entry point;
# dstl bootstraps kamping::v2 (and transitively mpi::core) from the sibling
# directories inside the fetched repo. kamping v1 and v2 coexist.
FetchContent_Declare(
  dstl
  GIT_REPOSITORY https://github.com/kamping-site/kamping-v2
  GIT_TAG 09047f3
  SOURCE_SUBDIR dstl
  EXCLUDE_FROM_ALL SYSTEM)

# Make v1 available first so its kamping::types target is reused by v2's
# bootstrap rather than fetched/defined a second time.
FetchContent_MakeAvailable(CLI11 fmt kamping dstl)

add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE MPI::MPI_CXX CLI11::CLI11
                                        OpenMP::OpenMP_CXX fmt::fmt kamping::kamping)
target_compile_features(benchmark PRIVATE cxx_std_20)

# Single-threaded baseline: MPI_THREAD_SINGLE init via mpi-core, MPI_Alltoallv.
add_executable(benchmark_singlethreaded benchmark_singlethreaded.cpp)
target_link_libraries(
  benchmark_singlethreaded
  PRIVATE MPI::MPI_CXX CLI11::CLI11 OpenMP::OpenMP_CXX fmt::fmt kamping::kamping
          kamping::v2)
target_compile_features(benchmark_singlethreaded PRIVATE cxx_std_20)

# Multi-threaded: MPI_THREAD_MULTIPLE init via mpi-core, dstl flat alltoallv.
add_executable(benchmark_multithreaded benchmark_multithreaded.cpp)
target_link_libraries(
  benchmark_multithreaded
  PRIVATE MPI::MPI_CXX CLI11::CLI11 OpenMP::OpenMP_CXX fmt::fmt kamping::kamping
          dstl::dstl)
target_compile_features(benchmark_multithreaded PRIVATE cxx_std_20)
