// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef STANDALONE_CED_PROBLEM_H
#define STANDALONE_CED_PROBLEM_H

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include "../CED_schedule/Problems.h"

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <map>
#include <string>
#include <vector>

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace standalone_ced {

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
class CedProblem {
public:
  CedProblem(int cloud_count, int edge_count, int device_count,
             int task_count, int operations_per_job,
             const std::string& data_path, const std::string& power_path);
  ~CedProblem();

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  CedProblem(const CedProblem&) = delete;
  CedProblem& operator=(const CedProblem&) = delete;

  // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
  int dimension() const { return 2 * task_count_ + 2 * operation_count_; }
  int task_count() const { return task_count_; }
  int operation_count() const { return operation_count_; }
  int operations_per_job() const { return operations_per_job_; }
  double task_computation(int task) const {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return tasks_[task].Computation;
  }
  double task_communication(int task) const {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return tasks_[task].Communication;
  }
  int task_relation_count(int task) const {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return static_cast<int>(tasks_[task].Precedence.size() +
                            tasks_[task].Interact.size() +
                            tasks_[task].Start_Pre.size() +
                            tasks_[task].End_Pre.size());
  }
  int task_edge_choice_count(int task) const {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return static_cast<int>(tasks_[task].AvailEdgeServerList.size());
  }
  double operation_time(int operation) const {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return operation_times_[operation];
  }
  int operation_device_choice_count(int operation) const {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return static_cast<int>(eligible_devices_[operation].size());
  }
  std::vector<double> structured_initial_solution(int rule) const;
  double evaluate(const std::vector<double>& solution);
  CEDDetailedMetrics detailed_metrics(const std::vector<double>& solution);
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
  long long search_evaluation_count() const { return search_evaluation_count_; }
#endif

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
private:
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
  long long search_evaluation_count_ = 0;
#endif
  int cloud_count_;
  int edge_count_;
  int device_count_;
  int task_count_;
  int job_count_;
  int operations_per_job_;
  int operation_count_;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  std::vector<CETask> tasks_;
  std::vector<double> operation_times_;
  std::vector<std::vector<DistanceValue>> edge_device_storage_;
  std::vector<std::vector<DistanceValue>> device_device_storage_;
  std::vector<DistanceValue*> edge_device_rows_;
  std::vector<DistanceValue*> device_device_rows_;
  std::vector<std::vector<int>> eligible_devices_;
  std::vector<double> energy_;
  std::vector<std::vector<int>> cloud_devices_;
  std::vector<std::vector<int>> edge_devices_;
  std::vector<std::vector<int>> cloud_load_;
  std::vector<std::vector<int>> edge_load_;
  std::vector<std::vector<int>> device_load_;
  std::vector<std::vector<int>> coupled_devices_;
  std::vector<std::map<int, double>> edge_device_communication_;
  std::vector<std::vector<double>> start_storage_;
  std::vector<std::vector<double>> end_storage_;
  std::vector<double*> start_rows_;
  std::vector<double*> end_rows_;
  std::vector<double> task_start_;
  std::vector<double> task_end_;
  std::vector<double> edge_x_;
  std::vector<double> edge_y_;
  std::vector<double> device_x_;
  std::vector<double> device_y_;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  static DistanceValue encode_distance(double value);
  void load_data(const std::string& path);
  void load_power(const std::string& path);
};

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace standalone_ced

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
#endif
