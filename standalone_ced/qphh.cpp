/* QPHH CED adapter: state construction, Q-selected low-level scheduling
 * action, exact evaluation, reward/Q update, and greedy retention. */
#include "benchmark_config.h"
#include "ced_problem.h"
#include "mpi_support.h"

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
namespace sc = standalone_ced;

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
constexpr int kStates = 13;
constexpr int kActions = 3; // Effective value in the authors' QPHH.h.
constexpr double kEpsilon = 0.3;
constexpr double kDiscount = 0.5;
constexpr int kBestInsertionTrials = 15;

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
int state_from_change(double old_value, double new_value) {
  const double improvement = std::log(old_value) - std::log(new_value);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement < 0.0) return 0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement == 0.0) return 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1.0) return 11;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-1) return 2;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-2) return 3;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-3) return 4;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-4) return 5;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-5) return 6;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-6) return 7;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-7) return 8;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-8) return 9;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement > 1e-9) return 10;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 12;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
double reward_from_change(double old_value, double new_value) {
  const double improvement = std::log(old_value) - std::log(new_value);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement < 0.0) return -10.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement >= 1e-1) return 10.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement >= 1e-2) return 9.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement >= 1e-3) return 8.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement >= 1e-4) return 7.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement >= 1e-5) return 6.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (improvement >= 1e-6) return 5.0;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 0.0;
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
template <typename Rng>
void swap_action(std::vector<double>& child, Rng& rng, int begin, int count) {
  std::uniform_int_distribution<int> pick(0, count - 1);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int repeat = 0; repeat < 3; ++repeat) {
    int first = pick(rng), second = pick(rng);
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    while (second == first) second = pick(rng);
    std::swap(child[begin + first], child[begin + second]);
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
template <typename Rng>
void insertion_action(std::vector<double>& child, Rng& rng, int begin,
                      int count) {
  std::uniform_int_distribution<int> pick(0, count - 1);
  const int chosen[3] = {pick(rng), pick(rng), pick(rng)};
  const int targets[3] = {pick(rng), 0, count - 1};
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int move = 0; move < 3; ++move) {
    const double target = child[begin + targets[move]];
    const double direction = targets[move] == count - 1 ? 1.0 : -1.0;
    child[begin + chosen[move]] =
        std::max(0.0, std::min(1.0, target + direction * 1e-12));
  }
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
int main(int argc, char** argv) {
  // 控制说明：进入受保护的主流程，将输入或运行错误统一转为可读诊断。
  try {
    sc::MpiSession mpi(&argc, &argv);
    sc::require_eight_processes(mpi.size());
    const int rank = mpi.rank();
    sc::CedProblem problem(sc::kCloudCount, sc::kEdgeCount, sc::kDeviceCount,
                           TNUM, MOPT_NUM, sc::kDataPath, sc::kPowerPath);
    const int dimension = problem.dimension();
    const int sequence_begin = 2 * TNUM;
    const int sequence_count = TNUM * MOPT_NUM;
    std::mt19937_64 rng(sc::experiment_seed() +
                        static_cast<unsigned>(rank) * 130363U);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> parent(dimension), child;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (double& value : parent) value = uniform(rng);
    double parent_fitness = problem.evaluate(parent);
    std::vector<double> initial_fitness;
    sc::gather_scalar(parent_fitness, initial_fitness);
    sc::trace_convergence(rank, 0, initial_fitness);
    double global_best = *std::min_element(initial_fitness.begin(),
                                           initial_fitness.end());

    // 段落说明：执行 MPI 分区、同步、迁移或归约，使各子种群按统一并行语义协作。
    std::array<std::array<double, kActions>, kStates> q{};
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (auto& row : q) row.fill(1.0);
    int state = rank == 0 ? static_cast<int>(rng() % kStates) : 0;
    const double start = sc::wall_time();
    int completed_generations = 0;
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    const long long initial_exact_evaluations =
// 控制说明：选择当前编译配置对应的实现路径。
#if defined(INCLUDE_INITIAL_EXACT_EVALUATIONS) && \
    INCLUDE_INITIAL_EXACT_EVALUATIONS
        0;
// 控制说明：选择当前编译配置对应的实现路径。
#else
        sc::global_sum(problem.search_evaluation_count());
#endif
    while (completed_generations < MAXGEN &&
           sc::global_sum(problem.search_evaluation_count()) -
                   initial_exact_evaluations <
               SEARCH_EXACT_EVALUATION_BUDGET) {
      const int generation = completed_generations;
// 控制说明：选择当前编译配置对应的实现路径。
#else
    while (completed_generations < MAXGEN) {
      const int generation = completed_generations;
#endif
      int action = 0;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (rank == 0) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (uniform(rng) < kEpsilon)
          action = static_cast<int>(rng() % kActions);
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else
          action = static_cast<int>(std::max_element(q[state].begin(),
                                                     q[state].end()) -
                                    q[state].begin());
      }
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
      MPI_Bcast(&action, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      child = parent;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (action == 0) {
        swap_action(child, rng, sequence_begin, sequence_count);
      } else if (action == 1) {
        insertion_action(child, rng, sequence_begin, sequence_count);
      } else {
        std::uniform_int_distribution<int> pick(0, sequence_count - 1);
        const int chosen = pick(rng);
        std::vector<double> best_child = child;
        double best_value = parent_fitness;
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int attempt = 0; attempt < kBestInsertionTrials; ++attempt) {
          std::vector<double> candidate = parent;
          const int target = pick(rng);
          candidate[sequence_begin + chosen] =
              std::max(0.0, std::min(1.0,
                  candidate[sequence_begin + target] +
                  (uniform(rng) < 0.5 ? -1e-12 : 1e-12)));
          const double value = problem.evaluate(candidate);
          // 控制说明：依据目标值决定接受、最优更新或审计路径。
          if (value < best_value) {
            best_value = value;
            best_child.swap(candidate);
          }
        }
        child.swap(best_child);
      }
      const double child_fitness = problem.evaluate(child);
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (child_fitness < parent_fitness) {
        parent.swap(child);
        parent_fitness = child_fitness;
      }
      std::vector<double> generation_fitness;
      sc::gather_scalar(parent_fitness, generation_fitness);
      const double new_best = *std::min_element(generation_fitness.begin(),
                                                generation_fitness.end());
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (rank == 0) {
        const int next_state = state_from_change(global_best, new_best);
        const double reward = reward_from_change(global_best, new_best);
        const double alpha = 1.0 -
            0.9 * static_cast<double>(generation + 1) / MAXGEN;
        const double max_next = *std::max_element(q[next_state].begin(),
                                                  q[next_state].end());
        q[state][action] += alpha *
            (reward + kDiscount * max_next - q[state][action]);
        state = next_state;
        global_best = std::min(global_best, new_best);
      }
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
      MPI_Bcast(&global_best, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      MPI_Bcast(&state, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      ++completed_generations;
      sc::trace_convergence(rank, completed_generations, generation_fitness);
    }
    const double elapsed = sc::wall_time() - start;
    const CEDDetailedMetrics detailed = problem.detailed_metrics(parent);
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    const long long search_exact_evaluations =
        sc::global_sum(problem.search_evaluation_count()) -
        initial_exact_evaluations;
#endif
    if (rank == 0) {
      std::cout << "Algorithm = QPHH (standalone CED mapping, NP=8)\n"
                << "Generation = " << completed_generations << '\n'
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
                << "Search exact evaluations = "
                << search_exact_evaluations
                << '\n'
#endif
                << "The best solution = " << global_best << '\n'
                << "Time = " << elapsed << " s\n";
    }
    sc::print_global_metrics(parent_fitness, detailed);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 0;
}
