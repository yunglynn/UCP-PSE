#ifndef STANDALONE_CED_MPI_SUPPORT_H
#define STANDALONE_CED_MPI_SUPPORT_H

#include "../CED_schedule/Problems.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>
#include <iostream>

namespace standalone_ced {

class MpiSession {
public:
  MpiSession(int* argc, char*** argv) {
#ifdef USE_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &size_);
#endif
  }
  ~MpiSession() {
#ifdef USE_MPI
    MPI_Finalize();
#endif
  }
  int rank() const { return rank_; }
  int size() const { return size_; }

private:
  int rank_ = 0;
  int size_ = 1;
};

inline double wall_time() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

inline void require_eight_processes(int size) {
  if (size != 8)
    throw std::runtime_error("standalone comparison requires 8 MPI processes");
}

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

inline void print_global_metrics(double local_objective,
                                 CEDDetailedMetrics metrics) {
  int local_rank = 0;
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

inline void gather_vector(const std::vector<double>& local,
                          std::vector<double>& global) {
#ifdef USE_MPI
  global.resize(local.size() * 8U);
  MPI_Allgather(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                global.data(), static_cast<int>(local.size()), MPI_DOUBLE,
                MPI_COMM_WORLD);
#else
  global = local;
#endif
}

inline void gather_scalar(double local, std::vector<double>& global) {
#ifdef USE_MPI
  global.resize(8);
  MPI_Allgather(&local, 1, MPI_DOUBLE, global.data(), 1, MPI_DOUBLE,
                MPI_COMM_WORLD);
#else
  global.assign(1, local);
#endif
}

inline void gather_int(int local, std::vector<int>& global) {
#ifdef USE_MPI
  global.resize(8);
  MPI_Allgather(&local, 1, MPI_INT, global.data(), 1, MPI_INT,
                MPI_COMM_WORLD);
#else
  global.assign(1, local);
#endif
}

inline void trace_convergence(int rank, int generation,
                              const std::vector<double>& fitness) {
#ifdef CONVERGENCE_TRACE
  if (rank == 0 && !fitness.empty())
    std::cout << "CONVERGENCE," << generation << ','
              << *std::min_element(fitness.begin(), fitness.end()) << '\n';
#else
  (void)rank;
  (void)generation;
  (void)fitness;
#endif
}

#ifdef SEARCH_EXACT_EVALUATION_BUDGET
inline long long global_sum(long long local) {
#ifdef USE_MPI
  long long global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  return global;
#else
  return local;
#endif
}
#endif

class SharedDoubleBuffer {
public:
  explicit SharedDoubleBuffer(size_t count) : count_(count) {
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
#else
    storage_.resize(count);
    data_ = storage_.data();
#endif
  }
  ~SharedDoubleBuffer() {
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
#ifdef USE_MPI
    MPI_Win_sync(window_);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Win_sync(window_);
#endif
  }

private:
  static int rank() {
#ifdef USE_MPI
    int value = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &value);
    return value;
#else
    return 0;
#endif
  }
  size_t count_ = 0;
  double* data_ = nullptr;
#ifdef USE_MPI
  MPI_Win window_ = MPI_WIN_NULL;
#else
  std::vector<double> storage_;
#endif
};

} // namespace standalone_ced

#endif
