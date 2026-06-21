#include "reporting.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <kamping/communicator.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/nlohmann_json_adapter/printer.hpp>

auto make_output_stream(std::string const& output_file)
    -> std::unique_ptr<std::ostream> {
  if (output_file == "stdout") {
    return std::make_unique<std::ostream>(std::cout.rdbuf());
  }
  if (output_file == "stderr") {
    return std::make_unique<std::ostream>(std::cerr.rdbuf());
  }
  auto path = std::filesystem::path(output_file);
  if (path.extension() != ".json") {
    path.replace_extension(".json");
  }
  auto file_stream = std::make_unique<std::ofstream>(path);
  if (!file_stream->is_open()) {
    throw std::runtime_error("Failed to open output file: " + path.string());
  }
  return file_stream;
}

void Report::step_iteration() {
  kamping::measurements::NLohmannJsonPrinter time_printer;
  kamping::measurements::timer().aggregate_and_print(time_printer);
  if (kamping::comm_world().is_root()) {
    times_.emplace_back(time_printer.json());
  }
  kamping::measurements::timer().clear();

  using counter_type = typename std::remove_reference_t<
      decltype(kamping::measurements::counter())>::DataType;
  kamping::measurements::NLohmannJsonPrinter<counter_type> counter_printer;
  kamping::measurements::counter().aggregate_and_print(counter_printer);
  if (kamping::comm_world().is_root()) {
    counters_.emplace_back(counter_printer.json());
  }
  kamping::measurements::counter().clear();
}

void Report::print(std::ostream& out) {
  if (kamping::comm_world().is_root()) {
    nlohmann::ordered_json json;
    json["config"] = config_;
    json["stats"] = stats_;
    json["timer"] = times_;
    json["counters"] = counters_;
    out << json.dump(2) << '\n';
  }
}
