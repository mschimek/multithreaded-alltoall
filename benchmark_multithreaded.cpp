#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <mpi.h>
#include <omp.h>
#include <CLI/CLI.hpp>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// kamping v1: timer / measurement infrastructure (kept for the benchmark output).
#include <kamping/measurements/printer.hpp>
#include <kamping/measurements/timer.hpp>

// kamping v2 / dSTL: RAII MPI initialization, communicator + collectives, and the
// thread_multiple_comm + flat dstl::alltoallv exchange.
#include "dstl/flat_alltoallv.hpp"
#include "dstl/thread_multiple_comm.hpp"
#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/collectives/barrier.hpp"
#include "mpi/comm.hpp"
#include "mpi/environment.hpp"

#include "./benchmark_check.hpp"
#include "./benchmark_common.hpp"

int main(int argc, char* argv[]) {
  CLI::App app;
  std::size_t num_threads = 1;
  app.add_option("-p,--threads", num_threads);
  std::size_t num_elements_per_worker = 32000;
  app.add_option("-n,--num-elements-per-worker", num_elements_per_worker);
  std::size_t iterations = 10;
  app.add_option("-i,--iterations", iterations);
  std::string distribution = "uniform";
  app.add_option("-d,--distribution", distribution, "uniform|nonuniform");
  std::uint64_t seed = 42;
  app.add_option("--seed", seed);
  CLI11_PARSE(app, argc, argv);
  Distribution dist = parse_distribution(distribution);

  // The dstl thread_multiple exchange issues concurrent MPI calls from several
  // threads, so it requires MPI_THREAD_MULTIPLE.
  mpi::experimental::environment env(argc, argv,
                                     mpi::experimental::ThreadLevel::multiple);

  mpi::experimental::comm_view world{MPI_COMM_WORLD};

  mpi::experimental::barrier(world);
  if (world.rank() == 0) {
    fmt::println(
        "thread level provided {}",
        static_cast<int>(mpi::experimental::environment::thread_level()));
  }

  // OpenMP setup
  omp_set_dynamic(0);
  omp_set_num_threads(static_cast<int>(num_threads));

  std::vector<std::pair<std::string, std::string>> config;
  config.emplace_back("threads", std::to_string(num_threads));
  config.emplace_back("elements", std::to_string(num_elements_per_worker));
  config.emplace_back("iterations", std::to_string(iterations));
  config.emplace_back("distribution", to_string(dist));
  config.emplace_back("seed", std::to_string(seed));

  auto [send_buf, send_counts, send_displs] =
      generate_data(world, num_threads, num_elements_per_worker, dist, seed);
  std::size_t total_buffer_size = send_buf.size();
  kamping::v2::allreduce(kamping::v2::inplace,
                         kamping::v2::views::ref_single(total_buffer_size),
                         std::plus<>{}, world);
  mpi::experimental::barrier(world);

  if (world.rank() == 0) {
    fmt::println(
        "Finished data generation with {} elements per worker and {} elements "
        "in total",
        send_buf.size(), total_buffer_size);
  }

  // Oracle: the recv buffer a correct alltoallv must produce (computed once).
  std::vector<int> expected = reference_alltoallv(
      MPI_COMM_WORLD, send_buf.data(), send_counts, send_displs);

  namespace views = kamping::v2::views;
  // The dstl thread_multiple_comm owns omp_get_max_threads() duplicated
  // communicators; build it once (the dup is collective) and reuse it.
  dstl::thread_multiple_comm fc{mpi::experimental::comm_view{MPI_COMM_WORLD}};
  for (std::size_t iteration = 0; iteration < iterations; iteration++) {
    std::vector<int> recv_buf;
    kamping::measurements::timer().synchronize_and_start(
        "multi-threaded-alltoall");
    dstl::alltoallv(send_buf | views::with_counts(send_counts) |
                        views::with_displs(send_displs),
                    recv_buf | views::auto_recv_v, fc);
    kamping::measurements::timer().stop_and_append();
    if (!matches_reference(recv_buf, expected)) {
      std::cerr << "Invalid result!\n";
    }
  }

  std::stringstream sstr;
  kamping::measurements::SimpleJsonPrinter<double> printer_timer(sstr, config);
  kamping::measurements::timer().aggregate_and_print(printer_timer);

  if (world.rank() == 0) {
    std::cout << sstr.str() << std::endl;
  }

  return 0;
}
