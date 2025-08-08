#include "./default_init_allocator.hpp"
#include <CLI/CLI.hpp>
#include <algorithm>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <mpi.h>
#include <numeric>
#include <omp.h>

int main(int argc, char *argv[]) {
  int thread_level = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_level);
  if (thread_level < MPI_THREAD_MULTIPLE) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  CLI::App app;
  std::size_t num_threads = 1;
  app.add_option("-p,--threads", num_threads);
  std::size_t num_elements = 32000;
  app.add_option("-n", num_elements);
  CLI11_PARSE(app, argc, argv);
  omp_set_num_threads(num_threads);

  std::vector<MPI_Comm> thread_comm(num_threads);
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  for (auto &comm : thread_comm) {
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  }
  std::vector<int, kamping::default_init_allocator<int>> send_buf(num_elements);

  std::vector<int> send_counts(size);
  std::vector<std::vector<int>> thread_send_counts(num_threads,
                                                   std::vector<int>(size));
  std::vector<int> send_displs(size);
  std::vector<std::vector<int>> thread_send_displs(num_threads,
                                                   std::vector<int>(size));

  std::vector<int> recv_counts(size);
  std::vector<std::vector<int>> thread_recv_counts(num_threads,
                                                   std::vector<int>(size));
  std::vector<int> recv_displs(size);
  std::vector<std::vector<int>> thread_recv_displs(num_threads,
                                                   std::vector<int>(size));

  auto elems_per_rank = num_elements / size;
  auto remainder = num_elements % size;
  for (std::size_t dest = 0; dest < static_cast<std::size_t>(size); dest++) {
    send_counts[dest] = elems_per_rank + (dest < remainder);
    if (dest == 0) {
      send_displs[dest] = 0;
    } else {
      send_displs[dest] = send_displs[dest - 1] + send_counts[dest - 1];
    }
    auto elems_per_thread = send_counts[dest] / num_threads;
    auto remainder_threads = send_counts[dest] % num_threads;
    for (std::size_t tid = 0; tid < num_threads; tid++) {
      thread_send_counts[tid][dest] =
          elems_per_thread + (tid < remainder_threads);
      if (tid == 0) {
        thread_send_displs[tid][dest] = send_displs[dest];
      } else {
        thread_send_displs[tid][dest] = thread_send_displs[tid - 1][dest] +
                                        thread_send_counts[tid - 1][dest];
      }
    }
  }

  // fill the send buffer
#pragma omp parallel for schedule(static)
  for (std::size_t dest = 0; dest < static_cast<std::size_t>(size); dest++) {
    std::fill_n(send_buf.begin() + send_displs[dest], send_counts[dest], dest);
  }

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    MPI_Alltoall(thread_send_counts[tid].data(), 1, MPI_INT,
                 thread_recv_counts[tid].data(), 1, MPI_INT, thread_comm[tid]);
  }
#pragma omp parallel for schedule(static)
  for (std::size_t source = 0; source < static_cast<std::size_t>(size);
       source++) {
    for (std::size_t tid = 0; tid < num_threads; tid++) {
      recv_counts[source] += thread_recv_counts[tid][source];
    }
  }

  std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                      recv_displs.begin(), 0);

#pragma omp parallel for schedule(static)
  for (std::size_t source = 0; source < static_cast<std::size_t>(size);
       source++) {
    for (std::size_t tid = 0; tid < num_threads; tid++) {
      if (tid == 0) {
        thread_recv_displs[tid][source] = recv_displs[source];
      } else {
        thread_recv_displs[tid][source] = thread_recv_displs[tid - 1][source] +
                                          thread_recv_counts[tid - 1][source];
      }
    }
  }
  // fmt::println("send_counts={}, send_displs={}", send_counts, send_displs);
  // fmt::println("recv_counts={}, recv_displs={}", recv_counts, recv_displs);
  // fmt::println("thread_send_counts={}, thread_send_displs={}",
  //              thread_send_counts, thread_send_displs);
  // fmt::println("thread_recv_counts={}, thread_recv_displs={}",
  //              thread_recv_counts, thread_recv_displs);

  std::vector<int, kamping::default_init_allocator<int>> recv_buf(
      recv_displs.back() + recv_counts.back());

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    MPI_Alltoallv(send_buf.data(), thread_send_counts[tid].data(),
                  thread_send_displs[tid].data(), MPI_INT, recv_buf.data(),
                  thread_recv_counts[tid].data(),
                  thread_recv_displs[tid].data(), MPI_INT, thread_comm[tid]);
  }
  // fmt::println("recv_buf={}", recv_buf);

  bool all_correct = std::all_of(recv_buf.begin(), recv_buf.end(),
                                 [&](auto &val) { return val == rank; });
  if (!all_correct) {
    std::cerr << "Invalid result!\n";
  }

  for (auto &comm : thread_comm) {
    MPI_Comm_free(&comm);
  }
  MPI_Finalize();
  return 0;
}
