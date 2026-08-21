#ifndef STANDALONE_CED_PROBLEM_H
#define STANDALONE_CED_PROBLEM_H

#include "../CED_schedule/Problems.h"

#include <map>
#include <string>
#include <vector>

namespace standalone_ced {

class CedProblem {
public:
  CedProblem(int cloud_count, int edge_count, int device_count,
             int task_count, int operations_per_job,
             const std::string& data_path, const std::string& power_path);
  ~CedProblem();

  CedProblem(const CedProblem&) = delete;
  CedProblem& operator=(const CedProblem&) = delete;

  int dimension() const { return 2 * task_count_ + 2 * operation_count_; }
  int task_count() const { return task_count_; }
  int operation_count() const { return operation_count_; }
  int operations_per_job() const { return operations_per_job_; }
  double task_computation(int task) const {
    return tasks_[task].Computation;
  }
  double task_communication(int task) const {
    return tasks_[task].Communication;
  }
  int task_relation_count(int task) const {
    return static_cast<int>(tasks_[task].Precedence.size() +
                            tasks_[task].Interact.size() +
                            tasks_[task].Start_Pre.size() +
                            tasks_[task].End_Pre.size());
  }
  int task_edge_choice_count(int task) const {
    return static_cast<int>(tasks_[task].AvailEdgeServerList.size());
  }
  double operation_time(int operation) const {
    return operation_times_[operation];
  }
  int operation_device_choice_count(int operation) const {
    return static_cast<int>(eligible_devices_[operation].size());
  }
  std::vector<double> structured_initial_solution(int rule) const;
  double evaluate(const std::vector<double>& solution);
  CEDDetailedMetrics detailed_metrics(const std::vector<double>& solution);
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
  long long search_evaluation_count() const { return search_evaluation_count_; }
#endif

private:
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

  static DistanceValue encode_distance(double value);
  void load_data(const std::string& path);
  void load_power(const std::string& path);
};

} // namespace standalone_ced

#endif
