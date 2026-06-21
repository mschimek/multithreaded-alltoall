#pragma once

#include <concepts>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

// Open the report's output stream: "stdout"/"stderr" map to the standard
// streams, anything else is a file path (a ".json" extension is appended if
// missing). Mirrors the kascade reporting approach.
auto make_output_stream(std::string const& output_file)
    -> std::unique_ptr<std::ostream>;

// Collects per-iteration timer (and counter) measurements together with a
// config/stats header and prints them as one structured JSON document. Mirrors
// kascade's Report class.
class Report {
public:
  void step_iteration();
  void print(std::ostream& out);

  void push_stats(std::string const& key, std::invocable auto&& compute_stats) {
    if (stats_.contains(key)) {
      return;
    }
    stats_[key] = compute_stats();
  }

  template <typename T>
  void push_stats(std::string const& key, T const& stats) {
    push_stats(key, [&]() { return stats; });
  }

  template <typename T>
  void push_config(T const& config) {
    config_ = config;
  }

private:
  nlohmann::ordered_json config_;
  nlohmann::ordered_json stats_;
  std::vector<nlohmann::ordered_json> times_;
  std::vector<nlohmann::ordered_json> counters_;
};
