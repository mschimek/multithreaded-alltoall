#pragma once

#include <mpi.h>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

// Correctness oracle for the alltoallv benchmarks.
//
// Both benchmarked exchanges (the single-threaded MPI_Alltoallv baseline and the
// multi-threaded dstl::alltoallv) must reproduce, byte-for-byte, the result of a
// plain MPI_Alltoallv on the same send data. We therefore redistribute the data
// once with a stock MPI_Alltoallv and compare each result against it. This is
// trivially true for the single-threaded baseline (it *is* an MPI_Alltoallv), but
// it is a meaningful validation of the multi-threaded dstl implementation.

// Reference recv buffer produced by a plain MPI_Alltoallv on the given send data.
inline std::vector<int> reference_alltoallv(MPI_Comm comm,
                                            int const* send_buf,
                                            std::vector<int> const& send_counts,
                                            std::vector<int> const& send_displs) {
  int size = 0;
  MPI_Comm_size(comm, &size);

  std::vector<int> recv_counts(static_cast<std::size_t>(size));
  std::vector<int> recv_displs(static_cast<std::size_t>(size));
  MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT,
               comm);
  std::exclusive_scan(recv_counts.begin(), recv_counts.end(),
                      recv_displs.begin(), 0);

  std::vector<int> recv_buf(
      static_cast<std::size_t>(recv_displs.back() + recv_counts.back()));
  MPI_Alltoallv(send_buf, send_counts.data(), send_displs.data(), MPI_INT,
                recv_buf.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                comm);
  return recv_buf;
}

// Element-wise comparison of an alltoallv result against the reference. The
// four-iterator std::equal also catches length mismatches, and the template
// tolerates differing container/allocator types between result and reference.
template <typename Result>
inline bool matches_reference(Result const& result,
                              std::vector<int> const& reference) {
  return std::equal(result.begin(), result.end(), reference.begin(),
                    reference.end());
}
