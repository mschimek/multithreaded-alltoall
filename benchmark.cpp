#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <mpi.h>
#include <mpi_proto.h>
#include <omp.h>
#include <CLI/CLI.hpp>
#include <algorithm>
#include <numeric>
#include "./default_init_allocator.hpp"

template <typename T>
using vec = std::vector<int, kamping::default_init_allocator<int>>;

class SingleThreadedAlltoAll {
public:
  SingleThreadedAlltoAll() {
    MPI_Info info = MPI_INFO_NULL;
    MPI_Info_create(&info);
    MPI_Info_set(info, "thread_level", "MPI_THREAD_FUNNELED");
    MPI_Session_init(info, MPI_ERRORS_ARE_FATAL, &session_);
    MPI_Info_free(&info);
    MPI_Group group = MPI_GROUP_NULL;
    MPI_Group_from_session_pset(session_, "mpi://WORLD", &group);
    MPI_Comm_create_from_group(group, "edu.kit.iti.ae.a2a.single",
                               MPI_INFO_NULL, MPI_ERRORS_ARE_FATAL, &comm_);
    MPI_Group_free(&group);
  }

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
    MPI_Alltoallv(send_buf.data(), send_counts.data(), send_displs.data(),
                  MPI_INT, recv_buf.data(), recv_counts.data(),
                  recv_displs.data(), MPI_INT, comm_);

    return recv_buf;
  }

  ~SingleThreadedAlltoAll() {
    MPI_Comm_free(&comm_);
    MPI_Session_finalize(&session_);
  }

private:
  MPI_Session session_ = MPI_SESSION_NULL;
  MPI_Comm comm_ = MPI_COMM_NULL;
};

class MultiThreadedAlltoAll {
public:
  MultiThreadedAlltoAll() {
    MPI_Info info = MPI_INFO_NULL;
    MPI_Info_create(&info);
    MPI_Info_set(info, "thread_level", "MPI_THREAD_MULTIPLE");
    MPI_Session_init(info, MPI_ERRORS_ARE_FATAL, &session_);
    MPI_Info_free(&info);
    MPI_Group group = MPI_GROUP_NULL;
    MPI_Group_from_session_pset(session_, "mpi://WORLD", &group);
    MPI_Comm_create_from_group(group, "edu.kit.iti.ae.a2a.single",
                               MPI_INFO_NULL, MPI_ERRORS_ARE_FATAL, &comm_);
    MPI_Group_free(&group);
    thread_comm_.resize(omp_get_max_threads());
    for (auto& comm : thread_comm_) {
      MPI_Comm_dup(comm_, &comm);
    }
  }

  vec<int> alltoall(vec<int> const& send_buf,
                    std::vector<int> const& send_counts,
                    std::vector<int> const& send_displs) const {
    int size = 0;
    MPI_Comm_size(comm_, &size);

    std::size_t max_threads = omp_get_max_threads();
    std::vector<std::vector<int>> thread_send_counts(max_threads,
                                                     std::vector<int>(size));
    std::vector<std::vector<int>> thread_send_displs(max_threads,
                                                     std::vector<int>(size));
    std::vector<std::vector<int>> thread_recv_counts(max_threads,
                                                     std::vector<int>(size));
    std::vector<std::vector<int>> thread_recv_displs(max_threads,
                                                     std::vector<int>(size));
    std::vector<int> recv_counts(size);
    std::vector<int> recv_displs(size);
    vec<int> recv_buf;

#pragma omp parallel
    {
      std::size_t num_threads = omp_get_num_threads();
      std::size_t my_tid = omp_get_thread_num();

      // compute thread-local send counts and displacements
#pragma omp for schedule(static) nowait
      for (std::size_t dest = 0; dest < static_cast<std::size_t>(size);
           dest++) {
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
#pragma omp barrier
      // exchange the counts
      MPI_Alltoall(thread_send_counts[my_tid].data(), 1, MPI_INT,
                   thread_recv_counts[my_tid].data(), 1, MPI_INT,
                   thread_comm_[my_tid]);
#pragma omp for schedule(static)
      for (std::size_t source = 0; source < static_cast<std::size_t>(size);
           source++) {
        for (std::size_t tid = 0; tid < num_threads; tid++) {
          recv_counts[source] += thread_recv_counts[tid][source];
        }
      }

#pragma omp single
      {
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                            recv_displs.begin(), 0);
      }
#pragma omp for schedule(static)
      for (std::size_t source = 0; source < static_cast<std::size_t>(size);
           source++) {
        for (std::size_t tid = 0; tid < num_threads; tid++) {
          if (tid == 0) {
            thread_recv_displs[tid][source] = recv_displs[source];
          } else {
            thread_recv_displs[tid][source] =
                thread_recv_displs[tid - 1][source] +
                thread_recv_counts[tid - 1][source];
          }
        }
      }
#pragma omp single
      {
        recv_buf.resize(recv_displs.back() + recv_counts.back());
      }
      MPI_Alltoallv(send_buf.data(), thread_send_counts[my_tid].data(),
                    thread_send_displs[my_tid].data(), MPI_INT, recv_buf.data(),
                    thread_recv_counts[my_tid].data(),
                    thread_recv_displs[my_tid].data(), MPI_INT,
                    thread_comm_[my_tid]);
    }

    return recv_buf;
  }

  ~MultiThreadedAlltoAll() {
    for (auto& comm : thread_comm_) {
      MPI_Comm_free(&comm);
    }
    MPI_Comm_free(&comm_);
    MPI_Session_finalize(&session_);
  }

private:
  MPI_Session session_ = MPI_SESSION_NULL;
  MPI_Comm comm_ = MPI_COMM_NULL;
  std::vector<MPI_Comm> thread_comm_;
};

auto generate_data(MPI_Comm comm, std::size_t num_elements) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  vec<int> send_buf(num_elements);
  std::vector<int> send_counts(size);
  std::vector<int> send_displs(size);

  auto elems_per_rank = num_elements / size;
  auto remainder = num_elements % size;
#pragma omp parallel
  {
#pragma omp for schedule(static)
    for (std::size_t dest = 0; dest < static_cast<std::size_t>(size); dest++) {
      send_counts[dest] = elems_per_rank + (dest < remainder);
    }
#pragma omp single
    {
      std::exclusive_scan(send_counts.begin(), send_counts.end(),
                          send_displs.begin(), 0);
    }
    // fill the send buffer
#pragma omp for schedule(static)
    for (std::size_t dest = 0; dest < static_cast<std::size_t>(size); dest++) {
      std::fill_n(send_buf.begin() + send_displs[dest], send_counts[dest],
                  dest);
    }
  }

  return std::make_tuple(std::move(send_buf), std::move(send_counts),
                         std::move(send_displs));
}

int main(int argc, char* argv[]) {
  int thread_level = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_level);
  CLI::App app;
  std::size_t num_threads = 1;
  app.add_option("-p,--threads", num_threads);
  std::size_t num_elements = 32000;
  app.add_option("-n,--num_elements", num_elements);
  std::size_t iterations = 10;
  app.add_option("-i,--iterations", iterations);
  CLI11_PARSE(app, argc, argv);
  omp_set_dynamic(0);
  omp_set_num_threads(num_threads);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  auto [send_buf, send_counts, send_displs] =
      generate_data(MPI_COMM_WORLD, num_elements);

  fmt::println("Finished data generation");

  // {
  //   SingleThreadedAlltoAll a2a_single;
  //   for (std::size_t iteration = 0; iteration < iterations; iteration++) {
  //     fmt::println("Starting iteration {}", iteration);
  //     auto recv_buf = a2a_single.alltoall(send_buf, send_counts, send_displs);
  //     bool all_correct = std::all_of(recv_buf.begin(), recv_buf.end(),
  //                                    [&](auto& val) { return val == rank; });
  //     if (!all_correct) {
  //       std::cerr << "Invalid result!\n";
  //     }
  //   }
  // }
  // fmt::println("Done with single threaded");

  {
    MultiThreadedAlltoAll a2a_multi;
    for (std::size_t iteration = 0; iteration < iterations; iteration++) {
      fmt::println("Starting iteration {}", iteration);
      auto recv_buf = a2a_multi.alltoall(send_buf, send_counts, send_displs);
      bool all_correct = std::all_of(recv_buf.begin(), recv_buf.end(),
                                     [&](auto& val) { return val == rank; });
      if (!all_correct) {
        std::cerr << "Invalid result!\n";
      }
    }
  }

  MPI_Finalize();
  return 0;
}
