// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include "ced_problem.h"

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace standalone_ced {
namespace {

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
template <typename T>
void read_value(std::istream& input, T& value, const char* field) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!(input >> value))
    throw std::runtime_error(std::string("invalid CED data while reading ") +
                             field);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void read_index_vector(std::istream& input, std::vector<int>& values,
                       int upper_bound, const char* field) {
  int count = 0;
  read_value(input, count, field);
  values.clear();
  values.reserve(count > 0 ? static_cast<size_t>(count) : 0U);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < count; ++i) {
    int value = 0;
    read_value(input, value, field);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (value >= 0 && value < upper_bound)
      values.push_back(value);
  }
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
std::vector<double> CedProblem::structured_initial_solution(int rule) const {
  rule = ((rule % 8) + 8) % 8;
  std::vector<double> x(dimension(), 0.0);
  const auto fractional = [](double value) {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return value - std::floor(value);
  };
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < task_count_; ++i) {
    const double comp = tasks_[i].Computation / 199.0;
    const double comm = tasks_[i].Communication / 4999.0;
    const double relation = task_relation_count(i) / 8.0;
    const double position = static_cast<double>(i) /
                            std::max(1, task_count_ - 1);
    // 控制说明：根据算法/动作枚举分派到对应实现，避免不同方法共享错误路径。
    switch (rule) {
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 0: x[i] = 0.0; break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 1: x[i] = 1.0; break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 2: x[i] = comp > comm ? 0.0 : 1.0; break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 3: x[i] = comp + relation > comm + 0.5 ? 0.0 : 1.0; break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 4: x[i] = 1.0 / (1.0 + std::exp(-4.0 * (comm - comp))); break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 5: x[i] = 1.0 / (1.0 + std::exp(-4.0 * (comp - comm))); break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
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
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < operation_count_; ++i) {
    const double duration = operation_times_[i] / 300.0;
    const double position = static_cast<double>(i) /
                            std::max(1, operation_count_ - 1);
    const double stage = static_cast<double>(i % operations_per_job_) /
                         std::max(1, operations_per_job_ - 1);
    double sequence = position;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (rule == 1) sequence = duration;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (rule == 2) sequence = 0.75 * stage + 0.25 * position;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (rule == 3) sequence = 0.75 * stage + 0.25 * duration;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (rule == 4) sequence = duration + stage;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (rule == 5) sequence = stage - duration;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (rule == 6) sequence = duration + 0.5 * position + stage;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (rule == 7) sequence = 1.0 - duration + stage;
    x[2 * task_count_ + i] = fractional(sequence);
    x[2 * task_count_ + operation_count_ + i] = fractional(
        duration + 0.38196601125 * position + 0.173 * rule);
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return x;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
DistanceValue CedProblem::encode_distance(double value) {
  const double scaled = value / DISTANCE_SCALE;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (scaled <= 0.0) return 0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (scaled >= 65535.0) return 65535;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return static_cast<DistanceValue>(scaled + 0.5);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
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
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < job_count_; ++i) {
    start_rows_[i] = start_storage_[i].data();
    end_rows_[i] = end_storage_[i].data();
  }
  load_data(data_path);
  load_power(power_path);
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
CedProblem::~CedProblem() = default;

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void CedProblem::load_data(const std::string& path) {
  std::ifstream input(path);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!input) throw std::runtime_error("cannot open CED data file: " + path);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  if (task_count_ >= 1000000) {
    std::string magic;
    read_value(input, magic, "compact header");
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (magic != "CED_COMPACT_V1")
      throw std::runtime_error("unexpected compact CED header");
    int edges = 0, devices = 0;
    read_value(input, edges, "compact edge count");
    read_value(input, devices, "compact device count");
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (edges != edge_count_ || devices != device_count_)
      throw std::runtime_error("compact CED geometry dimensions do not match");
    edge_x_.resize(edge_count_); edge_y_.resize(edge_count_);
    device_x_.resize(device_count_); device_y_.resize(device_count_);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < edge_count_; ++i) {
      read_value(input, edge_x_[i], "edge x");
      read_value(input, edge_y_[i], "edge y");
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
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
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < edge_count_; ++i) {
      edge_device_rows_[i] = edge_device_storage_[i].data();
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < device_count_; ++j) {
        double distance = 0.0;
        read_value(input, distance, "edge-device distance");
        edge_device_storage_[i][j] = encode_distance(distance);
      }
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < device_count_; ++i) {
      device_device_rows_[i] = device_device_storage_[i].data();
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < device_count_; ++j) {
        double distance = 0.0;
        read_value(input, distance, "device-device distance");
        device_device_storage_[i][j] = encode_distance(distance);
      }
    }
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (double& value : operation_times_)
    read_value(input, value, "manufacturing operation time");
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < task_count_; ++i) {
    read_value(input, tasks_[i].Computation, "task computation");
    read_value(input, tasks_[i].Communication, "task communication");
    read_index_vector(input, tasks_[i].Precedence, task_count_, "precedence");
    read_index_vector(input, tasks_[i].Interact, task_count_, "interaction");
    read_index_vector(input, tasks_[i].Start_Pre, task_count_, "start relation");
    read_index_vector(input, tasks_[i].End_Pre, task_count_, "end relation");
    read_value(input, tasks_[i].Job_Constraints, "job constraint");
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < operation_count_; ++i) {
    read_index_vector(input, eligible_devices_[i], device_count_,
                      "eligible device");
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (eligible_devices_[i].empty()) eligible_devices_[i].push_back(0);
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < task_count_; ++i) {
    read_index_vector(input, tasks_[i].AvailEdgeServerList, edge_count_,
                      "available edge server");
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (tasks_[i].AvailEdgeServerList.empty())
      tasks_[i].AvailEdgeServerList.push_back(0);
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void CedProblem::load_power(const std::string& path) {
  std::ifstream input(path);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!input) throw std::runtime_error("cannot open power file: " + path);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (double& value : energy_) read_value(input, value, "power coefficient");
}

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
double CedProblem::evaluate(const std::vector<double>& solution) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
  ++search_evaluation_count_;
#endif
  if (static_cast<int>(solution.size()) != dimension())
    throw std::invalid_argument("CED solution has an invalid dimension");
  // 控制说明：返回本阶段计算结果或状态码给调用方。
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

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
CEDDetailedMetrics CedProblem::detailed_metrics(
    const std::vector<double>& solution) {
  evaluate(solution);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return CED_LastDetailedMetrics();
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace standalone_ced
