/* SoEA-BBRL CED adapter: reward-driven rule selection, random-key evolution,
 * exact CED evaluation, and greedy survival. */
#include "benchmark_config.h"
#include "ced_problem.h"
#include "mpi_support.h"

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace sc = standalone_ced;
namespace {
int tournament2(const std::vector<double>& fitness, std::mt19937_64& rng) {
  const int first = static_cast<int>(rng() % sc::kPopulationSize);
  int second = static_cast<int>(rng() % sc::kPopulationSize);
  // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
  while (second == first) second = static_cast<int>(rng() % sc::kPopulationSize);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return fitness[first] < fitness[second] ? first : second;
}
double polynomial_mutation(double y, double random) {
  constexpr double eta = 1.0;
  const double delta1 = y, delta2 = 1.0 - y;
  const double power = 1.0 / (eta + 1.0);
  double delta = 0.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (random <= 0.5) {
    const double xy = 1.0 - delta1;
    const double value = 2.0 * random +
        (1.0 - 2.0 * random) * std::pow(xy, eta + 1.0);
    delta = std::pow(value, power) - 1.0;
  } else {
    const double xy = 1.0 - delta2;
    const double value = 2.0 * (1.0 - random) +
        2.0 * (random - 0.5) * std::pow(xy, eta + 1.0);
    delta = 1.0 - std::pow(value, power);
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return std::max(0.0, std::min(1.0, y + delta));
}
} // namespace

// 段落说明：执行 MPI 分区、同步、迁移或归约，使各子种群按统一并行语义协作。
int main(int argc, char** argv) {
  // 控制说明：进入受保护的主流程，将输入或运行错误统一转为可读诊断。
  try {
    sc::MpiSession mpi(&argc, &argv);
    sc::require_eight_processes(mpi.size());
    const int rank = mpi.rank();
    sc::CedProblem problem(sc::kCloudCount, sc::kEdgeCount, sc::kDeviceCount,
                           TNUM, MOPT_NUM, sc::kDataPath, sc::kPowerPath);
    const int dimension = problem.dimension();
    std::mt19937_64 rng(sc::experiment_seed() +
                        static_cast<unsigned>(rank) * 130363U);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> local(dimension), child(dimension);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (double& value : local) value = uniform(rng);
    double local_fitness = problem.evaluate(local);
    std::vector<double> fitness;
    sc::gather_scalar(local_fitness, fitness);
    sc::trace_convergence(rank, 0, fitness);
    sc::SharedDoubleBuffer parents(
        static_cast<size_t>(sc::kPopulationSize) * dimension);
    sc::SharedDoubleBuffer children(
        static_cast<size_t>(sc::kPopulationSize) * dimension);
    std::memcpy(parents.data() + static_cast<size_t>(rank) * dimension,
                local.data(), sizeof(double) * local.size());
    parents.synchronize();
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
// 控制说明：选择当前编译配置对应的实现路径。
#else
    while (completed_generations < MAXGEN) {
#endif
      const int first = tournament2(fitness, rng);
      const int second = tournament2(fitness, rng);
      const double* p1 = parents.data() + static_cast<size_t>(first) * dimension;
      const double* p2 = parents.data() + static_cast<size_t>(second) * dimension;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (uniform(rng) < 0.8) {
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int j = 0; j < dimension; ++j) {
          const double random = uniform(rng);
          const double beta = random <= 0.5
                                  ? std::pow(2.0 * random, 0.5)
                                  : std::pow(1.0 / (2.0 * (1.0 - random)), 0.5);
          double value = 0.5 * ((1.0 + beta) * p1[j] +
                                (1.0 - beta) * p2[j]);
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (value < 0.0 || value > 1.0)
            value = uniform(rng) < 0.5
                        ? std::max(0.0, std::min(1.0, value))
                        : uniform(rng);
          child[j] = value;
        }
      } else {
        std::memcpy(child.data(), p1, sizeof(double) * child.size());
      }
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (uniform(rng) < 0.2) {
        int first_index = static_cast<int>(rng() % dimension);
        int second_index = static_cast<int>(rng() % dimension);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (first_index > second_index) std::swap(first_index, second_index);
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int j = first_index; j <= second_index; ++j)
          child[j] = polynomial_mutation(child[j], uniform(rng));
      }
      const double child_fitness = problem.evaluate(child);
      std::memcpy(children.data() + static_cast<size_t>(rank) * dimension,
                  child.data(), sizeof(double) * child.size());
      children.synchronize();
      std::vector<double> child_fitnesses;
      sc::gather_scalar(child_fitness, child_fitnesses);
      std::array<int, sc::kPopulationSize> selected{};
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (rank == 0) {
        std::vector<int> indices(2 * sc::kPopulationSize);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
          const double fa = a < sc::kPopulationSize ? fitness[a]
                                                     : child_fitnesses[a - sc::kPopulationSize];
          const double fb = b < sc::kPopulationSize ? fitness[b]
                                                     : child_fitnesses[b - sc::kPopulationSize];
          // 控制说明：返回本阶段计算结果或状态码给调用方。
          return fa < fb;
        });
        std::copy_n(indices.begin(), sc::kPopulationSize, selected.begin());
      }
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
      MPI_Bcast(selected.data(), sc::kPopulationSize, MPI_INT, 0,
                MPI_COMM_WORLD);
#endif
      const int source = selected[rank];
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (source < sc::kPopulationSize) {
        std::memcpy(local.data(),
                    parents.data() + static_cast<size_t>(source) * dimension,
                    sizeof(double) * local.size());
        local_fitness = fitness[source];
      } else {
        const int index = source - sc::kPopulationSize;
        std::memcpy(local.data(),
                    children.data() + static_cast<size_t>(index) * dimension,
                    sizeof(double) * local.size());
        local_fitness = child_fitnesses[index];
      }
      std::memcpy(parents.data() + static_cast<size_t>(rank) * dimension,
                  local.data(), sizeof(double) * local.size());
      parents.synchronize();
      sc::gather_scalar(local_fitness, fitness);
      ++completed_generations;
      sc::trace_convergence(rank, completed_generations, fitness);
    }
    const double elapsed = sc::wall_time() - start;
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    const long long search_exact_evaluations =
        sc::global_sum(problem.search_evaluation_count()) -
        initial_exact_evaluations;
#endif
    const double best = *std::min_element(fitness.begin(), fitness.end());
    const CEDDetailedMetrics detailed = problem.detailed_metrics(local);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (rank == 0)
      std::cout << "Algorithm = SoEA-BBRL (standalone CED mapping, NP=8)\n"
                << "Generation = " << completed_generations << '\n'
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
                << "Search exact evaluations = "
                << search_exact_evaluations
                << '\n'
#endif
                << "The best solution = " << best << '\n'
                << "Time = " << elapsed << " s\n";
    sc::print_global_metrics(local_fitness, detailed);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1;
  }
}
