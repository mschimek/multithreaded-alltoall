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

// kamping v2 / mpi-core: RAII MPI initialization, communicator + collectives.
#include "kamping/v2/collectives/allreduce.hpp"
#include "kamping/v2/sentinels.hpp"
#include "kamping/v2/views.hpp"
#include "mpi/collectives/barrier.hpp"
#include "mpi/comm.hpp"
#include "mpi/environment.hpp"

#include "./benchmark_check.hpp"
#include "./benchmark_common.hpp"
#include "./reporting.hpp"

// Baseline: standard single-threaded MPI_Alltoallv.
class SingleThreadedAlltoAll {
public:
  SingleThreadedAlltoAll(MPI_Comm comm = MPI_COMM_WORLD) : comm_(comm) {}

  vec<int> alltoall(vec<int> const& send_buf,
                    std::vector<int> const& send_counts,
                    std::vector<int> const& send_displs) const {
    int size = 0;
    MPI_Comm_size(comm_, &size);

    std::vector<int> recv_counts(size);
    std::vector<int> recv_displs(size);

    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
                 comm_);
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                        recv_displs.begin(), 0);

    vec<int> recv_buf(recv_displs.back() + recv_counts.back());

    kamping::measurements::timer().synchronize_and_start("MPI_Alltoallv");
    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                  MPI_INT, recv_buf.data(), recv_counts.data(),
                  recv_displs.data(), MPI_INT, comm_);
    kamping::measurements::timer().stop_and_append();
    return recv_buf;
  }

private:
  MPI_Comm comm_;
};

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

  // Single-threaded MPI: only the main thread ever issues MPI calls.
  mpi::experimental::environment env(argc, argv,
                                     mpi::experimental::ThreadLevel::single);

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
  // report exactly that value. Here it only parallelizes data generation; the
  // exchange itself is single-threaded.
  omp_set_dynamic(0);
  std::size_t num_threads = static_cast<std::size_t>(omp_get_max_threads());

  nlohmann::ordered_json config;
  config["algorithm"] = "single-threaded-alltoallv";
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

  SingleThreadedAlltoAll a2a_single(MPI_COMM_WORLD);
  for (std::size_t iteration = 0; iteration < iterations; iteration++) {
    kamping::measurements::timer().synchronize_and_start(
        "single-threaded-alltoall");
    auto recv_buf = a2a_single.alltoall(send_buf, send_counts, send_displs);
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
