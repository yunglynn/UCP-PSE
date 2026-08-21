// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef STANDALONE_CED_MPI_SUPPORT_H
#define STANDALONE_CED_MPI_SUPPORT_H

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include "../CED_schedule/Problems.h"

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifdef USE_MPI
#include <mpi.h>
#endif

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <iostream>

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace standalone_ced {

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
class MpiSession {
public:
  MpiSession(int* argc, char*** argv) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
#endif
  }
  ~MpiSession() {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
    MPI_Finalize();
#endif
  }
  int rank() const { return rank_; }
  int size() const { return size_; }

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
private:
  int rank_ = 0;
  int size_ = 1;
};

// 段落说明：实现 `wall_time`：完成该函数负责的数据准备、算法步骤和状态返回。
inline double wall_time() {
  using Clock = std::chrono::steady_clock;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
inline void require_eight_processes(int size) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (size != 8)
    throw std::runtime_error("standalone comparison requires 8 MPI processes");
}

// 段落说明：输出可审计的运行信息、指标或错误原因。
inline void print_load_distribution(const char* name,
                                    const CEDLoadDistribution& load) {
  std::cout << name << " load: mean=" << load.mean
            << ", sd=" << load.standard_deviation
            << ", min=" << load.minimum << ", median=" << load.median
            << ", p95=" << load.percentile95 << ", max=" << load.maximum
            << ", Jain=" << load.jain_index
            << ", total=" << load.total_count
            << ", used=" << load.used_count
            << ", active-Jain=" << load.active_jain_index << '\n';
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
inline void print_global_metrics(double local_objective,
                                 CEDDetailedMetrics metrics) {
  int local_rank = 0;
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
  struct { double value; int rank; } local_pair{local_objective, local_rank},
                                      global_pair{local_objective, local_rank};
  MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                MPI_COMM_WORLD);
  MPI_Bcast(&metrics, static_cast<int>(sizeof(metrics)), MPI_BYTE,
            global_pair.rank, MPI_COMM_WORLD);
#endif
  if (local_rank != 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  std::cout << "Raw energy = " << metrics.energy << '\n'
            << "Makespan = " << metrics.makespan << '\n'
            << "Mean manufacturing-operation queue wait = "
            << metrics.average_operation_queue_wait << '\n'
            << "Mean task communication time = "
            << metrics.average_task_communication_time << '\n'
            << "Total transportation time = "
            << metrics.total_transportation_time << '\n';
  print_load_distribution("Cloud", metrics.cloud_load);
  print_load_distribution("Edge", metrics.edge_load);
  print_load_distribution("Industrial device", metrics.device_load);
  print_load_distribution("Factory busy time", metrics.factory_busy_time);
  print_load_distribution("Factory idle time", metrics.factory_idle_time);
  print_load_distribution("All-resource busy time",
                          metrics.all_resource_busy_time);
  print_load_distribution("All-resource idle time",
                          metrics.all_resource_idle_time);
  print_load_distribution("Cloud busy time", metrics.cloud_busy_time);
  print_load_distribution("Cloud idle time", metrics.cloud_idle_time);
  print_load_distribution("Edge busy time", metrics.edge_busy_time);
  print_load_distribution("Edge idle time", metrics.edge_idle_time);
  print_load_distribution("Industrial-device busy time",
                          metrics.device_busy_time);
  print_load_distribution("Industrial-device idle time",
                          metrics.device_idle_time);
}

// 段落说明：执行 MPI 分区、同步、迁移或归约，使各子种群按统一并行语义协作。
inline void gather_vector(const std::vector<double>& local,
                          std::vector<double>& global) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
  global.resize(local.size() * 8U);
  MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                global.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                MPI_COMM_WORLD);
// 控制说明：选择当前编译配置对应的实现路径。
#else
  global = local;
#endif
}

// 段落说明：执行 MPI 分区、同步、迁移或归约，使各子种群按统一并行语义协作。
inline void gather_scalar(double local, std::vector<double>& global) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
  global.resize(8);
  MPI_Allgather(&local, 1, MPI_DOUBLE, global.data(), 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
// 控制说明：选择当前编译配置对应的实现路径。
#else
  global.assign(1, local);
#endif
}

// 段落说明：执行 MPI 分区、同步、迁移或归约，使各子种群按统一并行语义协作。
inline void gather_int(int local, std::vector<int>& global) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
  global.resize(8);
  MPI_Allgather(&local, 1, MPI_INT, global.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
// 控制说明：选择当前编译配置对应的实现路径。
#else
  global.assign(1, local);
#endif
}

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
inline void trace_convergence(int rank, int generation,
                              const std::vector<double>& fitness) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef CONVERGENCE_TRACE
  if (rank == 0 && !fitness.empty())
    std::cout << "CONVERGENCE," << generation << ','
              << *std::min_element(fitness.begin(), fitness.end()) << '\n';
// 控制说明：选择当前编译配置对应的实现路径。
#else
  (void)rank;
  (void)generation;
  (void)fitness;
#endif
}

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
inline long long global_sum(long long local) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
  long long global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return global;
// 控制说明：选择当前编译配置对应的实现路径。
#else
  return local;
#endif
}
#endif

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
class SharedDoubleBuffer {
public:
  explicit SharedDoubleBuffer(size_t count) : count_(count) {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
    MPI_Aint bytes = rank() == 0
                         ? static_cast<MPI_Aint>(count * sizeof(double))
                         : 0;
    void* local = nullptr;
    MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL,
                            MPI_COMM_WORLD, &local, &window_);
    MPI_Aint queried_bytes = 0;
    int displacement = 0;
    void* base = nullptr;
    MPI_Win_shared_query(window_, 0, &queried_bytes, &displacement, &base);
    data_ = static_cast<double*>(base);
    MPI_Win_lock_all(0, window_);
// 控制说明：选择当前编译配置对应的实现路径。
#else
    storage_.resize(count);
    data_ = storage_.data();
#endif
  }
  ~SharedDoubleBuffer() {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
    MPI_Win_unlock_all(window_);
    MPI_Win_free(&window_);
#endif
  }
  SharedDoubleBuffer(const SharedDoubleBuffer&) = delete;
  SharedDoubleBuffer& operator=(const SharedDoubleBuffer&) = delete;
  double* data() { return data_; }
  const double* data() const { return data_; }
  size_t size() const { return count_; }
  void synchronize() {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
    MPI_Win_sync(window_);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(window_);
#endif
  }

// 段落说明：执行 MPI 分区、同步、迁移或归约，使各子种群按统一并行语义协作。
private:
  static int rank() {
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
    int value = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &value);
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return value;
// 控制说明：选择当前编译配置对应的实现路径。
#else
    return 0;
#endif
  }
  size_t count_ = 0;
  double* data_ = nullptr;
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
  MPI_Win window_ = MPI_WIN_NULL;
// 控制说明：选择当前编译配置对应的实现路径。
#else
  std::vector<double> storage_;
#endif
};

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace standalone_ced

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
#endif
