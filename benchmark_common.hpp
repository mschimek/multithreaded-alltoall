#pragma once

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include "mpi/comm.hpp"

#include "./default_init_allocator.hpp"

// Send/recv buffer type: a vector that skips zero-initialization on resize.
template <typename T>
using vec = std::vector<int, kamping::default_init_allocator<int>>;

// How the per-destination send counts are distributed. In both modes a PE sends a
// total of `num_elements_per_worker * num_threads` elements.
enum class Distribution {
  // Each destination receives (almost) the same number of elements.
  uniform,
  // The per-destination counts are sampled uniformly from the simplex
  // {x_i >= 0, sum x_i = X} via the "broken stick" method, yielding a skewed,
  // non-uniform distribution while keeping the per-PE total fixed.
  nonuniform
};

inline Distribution parse_distribution(std::string const& s) {
  if (s == "nonuniform" || s == "non-uniform" || s == "skewed") {
    return Distribution::nonuniform;
  }
  return Distribution::uniform;
}

inline std::string to_string(Distribution distribution) {
  return distribution == Distribution::nonuniform ? "nonuniform" : "uniform";
}

// Uniform split of `num_elements` over `size` destinations (remainder spread over
// the first ranks).
inline std::vector<int> compute_send_counts_uniform(int size,
                                                    std::size_t num_elements) {
  std::vector<int> send_counts(size);
  auto elems_per_rank =
      static_cast<int>(num_elements / static_cast<std::size_t>(size));
  auto remainder =
      static_cast<int>(num_elements % static_cast<std::size_t>(size));
  for (int dest = 0; dest < size; dest++) {
    send_counts[dest] = elems_per_rank + (dest < remainder);
  }
  return send_counts;
}

// Non-uniform counts via the "broken stick": draw `size - 1` uniform cut points in
// [0, X], sort them, and prepend 0 / append X; the gaps between consecutive cuts are
// uniform over the simplex {x_i >= 0, sum x_i = X}. Rounding the (monotone) cut
// positions to integers and taking differences keeps the counts non-negative and
// summing to exactly `num_elements`.
inline std::vector<int> compute_send_counts_nonuniform(int size,
                                                       std::size_t num_elements,
                                                       std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(0.0,
                                              static_cast<double>(num_elements));

  std::vector<double> cuts(static_cast<std::size_t>(size) + 1);
  cuts.front() = 0.0;
  cuts.back() = static_cast<double>(num_elements);
  for (int i = 1; i < size; i++) {
    cuts[static_cast<std::size_t>(i)] = dist(rng);
  }
  std::sort(cuts.begin() + 1, cuts.end() - 1);

  std::vector<int> send_counts(size);
  long long prev = 0;
  for (int dest = 0; dest < size; dest++) {
    long long next =
        (dest + 1 == size)
            ? static_cast<long long>(num_elements)
            : std::llround(cuts[static_cast<std::size_t>(dest) + 1]);
    send_counts[dest] = static_cast<int>(next - prev);
    prev = next;
  }
  return send_counts;
}

// Generate the send buffer / counts / displacements for the alltoallv benchmark.
// Each rank produces `num_elements_per_worker * num_threads` elements; the value
// written for a given destination is the destination rank, so each rank receives
// only its own rank id (which keeps the correctness check valid in either mode).
// The per-destination split is `uniform` or the skewed `nonuniform` "broken stick".
inline auto generate_data(mpi::experimental::comm_view comm,
                          std::size_t num_threads,
                          std::size_t num_elements_per_worker,
                          Distribution distribution = Distribution::uniform,
                          std::uint64_t seed = 42) {
  int rank = comm.rank();
  int size = comm.size();
  std::size_t num_elements = num_elements_per_worker * num_threads;

  std::vector<int> send_counts =
      distribution == Distribution::uniform
          ? compute_send_counts_uniform(size, num_elements)
          : compute_send_counts_nonuniform(
                size, num_elements, seed + static_cast<std::uint64_t>(rank));

  std::vector<int> send_displs(size);
  std::exclusive_scan(send_counts.begin(), send_counts.end(),
                      send_displs.begin(), 0);

  vec<int> send_buf(num_elements);
#pragma omp parallel for schedule(static)
  for (int dest = 0; dest < size; dest++) {
    std::fill_n(send_buf.begin() + send_displs[dest], send_counts[dest],
                dest);
  }

  return std::make_tuple(std::move(send_buf), std::move(send_counts),
                         std::move(send_displs));
}
