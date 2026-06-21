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
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// kamping v1: timer / measurement infrastructure (kept for the benchmark output).
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
#include "./reporting.hpp"

int main(int argc, char* argv[]) {
  CLI::App app;
  std::size_t num_elements_per_worker = 32000;
  app.add_option("-n,--num-elements-per-worker", num_elements_per_worker);
  std::size_t iterations = 10;
  app.add_option("-i,--iterations", iterations);
  std::string distribution = "uniform";
  app.add_option("-d,--distribution", distribution, "uniform|nonuniform");
  std::uint64_t seed = 42;
  app.add_option("--seed", seed);
  std::string output_file = "stdout";
  app.add_option("--output-file", output_file,
                 "Write the report JSON here ('stdout'/'stderr' or a path; a "
                 "'.json' extension is added to file paths)");
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

  // The number of OpenMP threads (and thus the per-rank data volume) is taken
  // from the environment (OMP_NUM_THREADS), which the kaval command template
  // sets to threads_per_rank. omp_set_dynamic(0) makes omp_get_max_threads()
  // report exactly that value.
  omp_set_dynamic(0);
  std::size_t num_threads = static_cast<std::size_t>(omp_get_max_threads());

  nlohmann::ordered_json config;
  config["algorithm"] = "dstl-multi-threaded-alltoallv";
  config["threads"] = num_threads;
  config["elements_per_worker"] = num_elements_per_worker;
  config["iterations"] = iterations;
  config["distribution"] = to_string(dist);
  config["seed"] = seed;

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

  Report report;
  report.push_stats("total_elements", total_buffer_size);

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
    report.step_iteration();
  }

  if (world.rank() == 0) {
    auto out = make_output_stream(output_file);
    report.push_config(config);
    report.print(*out);
  }

  return 0;
}
