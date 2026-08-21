#include "ced_problem.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace standalone_ced {
namespace {

template <typename T>
void read_value(std::istream& input, T& value, const char* field) {
  if (!(input >> value))
    throw std::runtime_error(std::string("invalid CED data while reading ") +
                             field);
}

void read_index_vector(std::istream& input, std::vector<int>& values,
                       int upper_bound, const char* field) {
  int count = 0;
  read_value(input, count, field);
  values.clear();
  values.reserve(count > 0 ? static_cast<size_t>(count) : 0U);
  for (int i = 0; i < count; ++i) {
    int value = 0;
    read_value(input, value, field);
    if (value >= 0 && value < upper_bound)
      values.push_back(value);
  }
}

} // namespace

std::vector<double> CedProblem::structured_initial_solution(int rule) const {
  rule = ((rule % 8) + 8) % 8;
  std::vector<double> x(dimension(), 0.0);
  const auto fractional = [](double value) {
    return value - std::floor(value);
  };
  for (int i = 0; i < task_count_; ++i) {
    const double comp = tasks_[i].Computation / 199.0;
    const double comm = tasks_[i].Communication / 4999.0;
    const double relation = task_relation_count(i) / 8.0;
    const double position = static_cast<double>(i) /
                            std::max(1, task_count_ - 1);
    switch (rule) {
      case 0: x[i] = 0.0; break;
      case 1: x[i] = 1.0; break;
      case 2: x[i] = comp > comm ? 0.0 : 1.0; break;
      case 3: x[i] = comp + relation > comm + 0.5 ? 0.0 : 1.0; break;
      case 4: x[i] = 1.0 / (1.0 + std::exp(-4.0 * (comm - comp))); break;
      case 5: x[i] = 1.0 / (1.0 + std::exp(-4.0 * (comp - comm))); break;
      case 6:
        x[i] = 1.0 / (1.0 + std::exp(-3.0 * (comm + relation - comp)));
        break;
      default: x[i] = ((i + rule) & 1) ? 1.0 : 0.0; break;
    }
    const double resource = rule % 2 == 0
                                ? comp + 0.5 * comm + relation
                                : comm + 0.5 * comp - relation;
    x[task_count_ + i] = fractional(
        resource + 0.61803398875 * position + 0.137 * rule);
  }
  for (int i = 0; i < operation_count_; ++i) {
    const double duration = operation_times_[i] / 300.0;
    const double position = static_cast<double>(i) /
                            std::max(1, operation_count_ - 1);
    const double stage = static_cast<double>(i % operations_per_job_) /
                         std::max(1, operations_per_job_ - 1);
    double sequence = position;
    if (rule == 1) sequence = duration;
    else if (rule == 2) sequence = 0.75 * stage + 0.25 * position;
    else if (rule == 3) sequence = 0.75 * stage + 0.25 * duration;
    else if (rule == 4) sequence = duration + stage;
    else if (rule == 5) sequence = stage - duration;
    else if (rule == 6) sequence = duration + 0.5 * position + stage;
    else if (rule == 7) sequence = 1.0 - duration + stage;
    x[2 * task_count_ + i] = fractional(sequence);
    x[2 * task_count_ + operation_count_ + i] = fractional(
        duration + 0.38196601125 * position + 0.173 * rule);
  }
  return x;
}

DistanceValue CedProblem::encode_distance(double value) {
  const double scaled = value / DISTANCE_SCALE;
  if (scaled <= 0.0) return 0;
  if (scaled >= 65535.0) return 65535;
  return static_cast<DistanceValue>(scaled + 0.5);
}

CedProblem::CedProblem(int cloud_count, int edge_count, int device_count,
                       int task_count, int operations_per_job,
                       const std::string& data_path,
                       const std::string& power_path)
    : cloud_count_(cloud_count), edge_count_(edge_count),
      device_count_(device_count), task_count_(task_count),
      job_count_(task_count), operations_per_job_(operations_per_job),
      operation_count_(task_count * operations_per_job), tasks_(task_count),
      operation_times_(operation_count_), eligible_devices_(operation_count_),
      energy_(11), cloud_devices_(cloud_count), edge_devices_(edge_count),
      cloud_load_(cloud_count), edge_load_(edge_count),
      device_load_(device_count), coupled_devices_(task_count),
      edge_device_communication_(edge_count),
      start_storage_(job_count_, std::vector<double>(operations_per_job_)),
      end_storage_(job_count_, std::vector<double>(operations_per_job_)),
      start_rows_(job_count_), end_rows_(job_count_), task_start_(task_count),
      task_end_(task_count) {
  for (int i = 0; i < job_count_; ++i) {
    start_rows_[i] = start_storage_[i].data();
    end_rows_[i] = end_storage_[i].data();
  }
  load_data(data_path);
  load_power(power_path);
}

CedProblem::~CedProblem() = default;

void CedProblem::load_data(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open CED data file: " + path);

  if (task_count_ >= 1000000) {
    std::string magic;
    read_value(input, magic, "compact header");
    if (magic != "CED_COMPACT_V1")
      throw std::runtime_error("unexpected compact CED header");
    int edges = 0, devices = 0;
    read_value(input, edges, "compact edge count");
    read_value(input, devices, "compact device count");
    if (edges != edge_count_ || devices != device_count_)
      throw std::runtime_error("compact CED geometry dimensions do not match");
    edge_x_.resize(edge_count_); edge_y_.resize(edge_count_);
    device_x_.resize(device_count_); device_y_.resize(device_count_);
    for (int i = 0; i < edge_count_; ++i) {
      read_value(input, edge_x_[i], "edge x");
      read_value(input, edge_y_[i], "edge y");
    }
    for (int i = 0; i < device_count_; ++i) {
      read_value(input, device_x_[i], "device x");
      read_value(input, device_y_[i], "device y");
    }
    CED_SetCompactGeometry(edge_x_.data(), edge_y_.data(), device_x_.data(),
                           device_y_.data(), edge_count_, device_count_);
  } else {
    edge_device_storage_.assign(
        edge_count_, std::vector<DistanceValue>(device_count_));
    device_device_storage_.assign(
        device_count_, std::vector<DistanceValue>(device_count_));
    edge_device_rows_.resize(edge_count_);
    device_device_rows_.resize(device_count_);
    for (int i = 0; i < edge_count_; ++i) {
      edge_device_rows_[i] = edge_device_storage_[i].data();
      for (int j = 0; j < device_count_; ++j) {
        double distance = 0.0;
        read_value(input, distance, "edge-device distance");
        edge_device_storage_[i][j] = encode_distance(distance);
      }
    }
    for (int i = 0; i < device_count_; ++i) {
      device_device_rows_[i] = device_device_storage_[i].data();
      for (int j = 0; j < device_count_; ++j) {
        double distance = 0.0;
        read_value(input, distance, "device-device distance");
        device_device_storage_[i][j] = encode_distance(distance);
      }
    }
  }

  for (double& value : operation_times_)
    read_value(input, value, "manufacturing operation time");
  for (int i = 0; i < task_count_; ++i) {
    read_value(input, tasks_[i].Computation, "task computation");
    read_value(input, tasks_[i].Communication, "task communication");
    read_index_vector(input, tasks_[i].Precedence, task_count_, "precedence");
    read_index_vector(input, tasks_[i].Interact, task_count_, "interaction");
    read_index_vector(input, tasks_[i].Start_Pre, task_count_, "start relation");
    read_index_vector(input, tasks_[i].End_Pre, task_count_, "end relation");
    read_value(input, tasks_[i].Job_Constraints, "job constraint");
  }
  for (int i = 0; i < operation_count_; ++i) {
    read_index_vector(input, eligible_devices_[i], device_count_,
                      "eligible device");
    if (eligible_devices_[i].empty()) eligible_devices_[i].push_back(0);
  }
  for (int i = 0; i < task_count_; ++i) {
    read_index_vector(input, tasks_[i].AvailEdgeServerList, edge_count_,
                      "available edge server");
    if (tasks_[i].AvailEdgeServerList.empty())
      tasks_[i].AvailEdgeServerList.push_back(0);
  }
}

void CedProblem::load_power(const std::string& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open power file: " + path);
  for (double& value : energy_) read_value(input, value, "power coefficient");
}

double CedProblem::evaluate(const std::vector<double>& solution) {
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
  ++search_evaluation_count_;
#endif
  if (static_cast<int>(solution.size()) != dimension())
    throw std::invalid_argument("CED solution has an invalid dimension");
  return CED_Schedule(
      const_cast<double*>(solution.data()), cloud_count_, edge_count_,
      device_count_, task_count_, job_count_, operations_per_job_,
      tasks_.data(), operation_times_.data(),
      edge_device_rows_.empty() ? nullptr : edge_device_rows_.data(),
      device_device_rows_.empty() ? nullptr : device_device_rows_.data(),
      eligible_devices_.data(), energy_.data(), cloud_devices_.data(),
      edge_devices_.data(), cloud_load_.data(), edge_load_.data(),
      device_load_.data(), coupled_devices_.data(),
      edge_device_communication_.data(), start_rows_.data(), end_rows_.data(),
      task_start_.data(), task_end_.data());
}

CEDDetailedMetrics CedProblem::detailed_metrics(
    const std::vector<double>& solution) {
  evaluate(solution);
  return CED_LastDetailedMetrics();
}

} // namespace standalone_ced
