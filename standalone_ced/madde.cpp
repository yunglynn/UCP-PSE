/* MadDE CED adapter: strategy mutation, bounded trial generation, exact
 * evaluation, greedy selection, and adaptive strategy/parameter memories. */
#include "benchmark_config.h"
#include "ced_problem.h"
#include "mpi_support.h"

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <unordered_map>

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
namespace sc = standalone_ced;

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {

// 段落说明：实现 `std::min`：完成该函数负责的数据准备、算法步骤和状态返回。
double clamp01(double value) { return std::max(0.0, std::min(1.0, value)); }

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
class VirtualMemory {
public:
  VirtualMemory(size_t size, double initial) : size_(size), initial_(initial) {}
  double get(size_t index) const {
    auto found = values_.find(index);
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return found == values_.end() ? initial_ : found->second;
  }
  void set(size_t index, double value) { values_[index] = value; }
  size_t size() const { return size_; }

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
private:
  size_t size_;
  double initial_;
  std::unordered_map<size_t, double> values_;
};

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
int distinct_index(std::mt19937_64& rng, int a, int b = -1, int c = -1) {
  std::uniform_int_distribution<int> pick(0, sc::kPopulationSize - 1);
  int value = 0;
  // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
  do value = pick(rng); while (value == a || value == b || value == c);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return value;
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
double sample_cauchy_positive(std::mt19937_64& rng, double location) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  double value = 0.0;
  // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
  do value = location + 0.1 * std::tan(M_PI * (uniform(rng) - 0.5));
  // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
  while (value <= 0.0);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return std::min(1.0, value);
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
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
    std::normal_distribution<double> normal(0.0, 0.1);

    // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
    std::vector<double> local(dimension);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (double& value : local) value = uniform(rng);
    sc::SharedDoubleBuffer population_buffer(
        static_cast<size_t>(sc::kPopulationSize) * dimension);
    double* population = population_buffer.data();
    std::memcpy(population + static_cast<size_t>(rank) * dimension,
                local.data(), sizeof(double) * static_cast<size_t>(dimension));
    population_buffer.synchronize();
    double local_fitness = problem.evaluate(local);
    std::vector<double> fitness;
    sc::gather_scalar(local_fitness, fitness);
    sc::trace_convergence(rank, 0, fitness);

    // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
    constexpr double q_cr_rate = 0.01;
    constexpr double p_best_rate = 0.18;
    constexpr double archive_rate = 2.3;
    const int archive_limit = static_cast<int>(std::round(
        archive_rate * static_cast<double>(sc::kPopulationSize)));
    VirtualMemory memory_f(static_cast<size_t>(10) * dimension, 0.2);
    VirtualMemory memory_cr(static_cast<size_t>(10) * dimension, 0.2);
    size_t memory_position = 0;
    std::array<double, 3> operator_probability{{1.0 / 3.0, 1.0 / 3.0,
                                                1.0 / 3.0}};
    sc::SharedDoubleBuffer archive_buffer(
        static_cast<size_t>(archive_limit) * dimension);
    double* archive = archive_buffer.data();
    int archive_count = 0;
    sc::SharedDoubleBuffer trial_buffer(
        static_cast<size_t>(sc::kPopulationSize) * dimension);
    double* gathered_trials = trial_buffer.data();
    std::vector<double> trial(dimension), mutant(dimension);

    // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
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
      std::vector<int> order(sc::kPopulationSize);
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int i = 0; i < sc::kPopulationSize; ++i) order[i] = i;
      std::sort(order.begin(), order.end(), [&](int a, int b) {
          // 控制说明：返回本阶段计算结果或状态码给调用方。
          return fitness[a] < fitness[b];
      });

      // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
      std::uniform_int_distribution<size_t> memory_pick(0,
                                                        memory_f.size() - 1);
      const size_t memory_index = memory_pick(rng);
      const double mu_f = memory_f.get(memory_index);
      const double mu_cr = memory_cr.get(memory_index);
      const double scaling = sample_cauchy_positive(rng, mu_f);
      const double crossover = clamp01(mu_cr < 0.0 ? 0.0 : mu_cr + normal(rng));

      // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
      const double draw = uniform(rng);
      int op = draw <= operator_probability[0]
                   ? 0
                   : (draw <= operator_probability[0] + operator_probability[1]
                          ? 1
                          : 2);
      const int r1 = distinct_index(rng, rank);
      const int r2 = distinct_index(rng, rank, r1);
      const int r3 = distinct_index(rng, rank, r1, r2);
      std::uniform_int_distribution<int> archive_pick(
          0, sc::kPopulationSize + archive_count - 1);
      int r2_pool = archive_pick(rng);
      // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
      while (r2_pool < sc::kPopulationSize &&
             (r2_pool == rank || r2_pool == r1))
        r2_pool = archive_pick(rng);
      const int p_count = std::max(2, static_cast<int>(
          std::round(p_best_rate * sc::kPopulationSize)));
      std::uniform_int_distribution<int> p_pick(0, p_count - 1);
      const int pbest = order[p_pick(rng)];
      const double progress = static_cast<double>(generation * sc::kPopulationSize) /
                              std::max(1, MAXGEN * sc::kPopulationSize);
      const double q_rate = 2.0 * p_best_rate - p_best_rate * progress;
      const int q_count = std::max(2, static_cast<int>(
          std::round(q_rate * sc::kPopulationSize)));
      std::uniform_int_distribution<int> q_pick(0, q_count - 1);
      const int qbest = order[q_pick(rng)];
      const double attraction = 0.5 + 0.5 * progress;

      // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
      auto at = [&](int individual, int coordinate) -> double {
        // 控制说明：返回本阶段计算结果或状态码给调用方。
        return population[static_cast<size_t>(individual) * dimension + coordinate];
      };
      auto at_pool = [&](int individual, int coordinate) -> double {
        // 控制说明：返回本阶段计算结果或状态码给调用方。
        return individual < sc::kPopulationSize
                   ? at(individual, coordinate)
                   : archive[(static_cast<size_t>(individual - sc::kPopulationSize) *
                              dimension) + coordinate];
      };
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < dimension; ++j) {
        const double parent = at(rank, j);
        double value = parent;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (op == 0)
          value = parent + scaling *
              (at(pbest, j) - parent + at(r1, j) - at_pool(r2_pool, j));
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        else if (op == 1)
          value = parent + scaling * (at(r1, j) - at_pool(r2_pool, j));
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else
          value = scaling * (at(r1, j) + attraction *
                             (at(qbest, j) - at(r3, j)));
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (value < 0.0) value = 0.5 * parent;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (value > 1.0) value = 0.5 * (parent + 1.0);
        mutant[j] = value;
      }

      // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
      std::uniform_int_distribution<int> coordinate_pick(0, dimension - 1);
      const int forced = coordinate_pick(rng);
      const bool q_best_crossover = uniform(rng) <= q_cr_rate;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < dimension; ++j) {
        const double crossover_base = q_best_crossover ? at(qbest, j) : at(rank, j);
        trial[j] = (j == forced || uniform(rng) <= crossover)
                       ? mutant[j]
                       : crossover_base;
      }

      // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
      const double trial_fitness = problem.evaluate(trial);
      std::vector<double> trial_fitnesses, used_f, used_cr, improvements;
      std::vector<int> operators;
      sc::gather_scalar(trial_fitness, trial_fitnesses);
      sc::gather_scalar(scaling, used_f);
      sc::gather_scalar(crossover, used_cr);
      sc::gather_int(op, operators);
      std::memcpy(gathered_trials + static_cast<size_t>(rank) * dimension,
                  trial.data(), sizeof(double) * static_cast<size_t>(dimension));
      trial_buffer.synchronize();

      // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
      std::array<double, 3> operator_gain{{0.0, 0.0, 0.0}};
      double weighted_f_num = 0.0, weighted_f_den = 0.0;
      double weighted_cr_num = 0.0, weighted_cr_den = 0.0;
      bool any_success = false;
      std::vector<int> accepted(sc::kPopulationSize, 0);
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int i = 0; i < sc::kPopulationSize; ++i) {
        // 控制说明：依据目标值决定接受、最优更新或审计路径。
        if (trial_fitnesses[i] >= fitness[i]) continue;
        accepted[i] = 1;
        any_success = true;
        const double difference = fitness[i] - trial_fitnesses[i];
        const double relative = difference / (std::fabs(fitness[i]) + 1e-300);
        operator_gain[operators[i]] += std::max(0.0, relative);
        weighted_f_num += difference * used_f[i] * used_f[i];
        weighted_f_den += difference * used_f[i];
        weighted_cr_num += difference * used_cr[i] * used_cr[i];
        weighted_cr_den += difference * used_cr[i];
      }

      // 段落说明：执行 MPI 分区、同步、迁移或归约，使各子种群按统一并行语义协作。
      if (rank == 0) {
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int i = 0; i < sc::kPopulationSize; ++i) {
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (!accepted[i]) continue;
          int slot = archive_count;
          // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
          if (archive_count < archive_limit)
            ++archive_count;
          // 控制说明：条件不成立时执行互斥的备用处理路径。
          else {
            std::uniform_int_distribution<int> replace(0, archive_limit - 1);
            slot = replace(rng);
          }
          std::memcpy(archive + static_cast<size_t>(slot) * dimension,
                      population + static_cast<size_t>(i) * dimension,
                      sizeof(double) * static_cast<size_t>(dimension));
        }
      }
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef USE_MPI
      MPI_Bcast(&archive_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      archive_buffer.synchronize();
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (accepted[rank]) {
        std::memcpy(population + static_cast<size_t>(rank) * dimension,
                    gathered_trials + static_cast<size_t>(rank) * dimension,
                    sizeof(double) * static_cast<size_t>(dimension));
      }
      population_buffer.synchronize();
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int i = 0; i < sc::kPopulationSize; ++i)
        // 控制说明：依据目标值决定接受、最优更新或审计路径。
        if (accepted[i]) fitness[i] = trial_fitnesses[i];

      // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
      const double gain_sum = operator_gain[0] + operator_gain[1] + operator_gain[2];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (gain_sum > 0.0)
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int k = 0; k < 3; ++k)
          operator_probability[k] =
              std::max(0.1, std::min(0.9, operator_gain[k] / gain_sum));
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        operator_probability = {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}};

      // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
      if (any_success && weighted_f_den > 0.0) {
        memory_f.set(memory_position, weighted_f_num / weighted_f_den);
        memory_cr.set(memory_position,
                      weighted_cr_den > 0.0 ? weighted_cr_num / weighted_cr_den
                                            : -1.0);
      } else {
        memory_f.set(memory_position, 0.5);
        memory_cr.set(memory_position, 0.5);
      }
      memory_position = (memory_position + 1) % memory_f.size();
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
    std::memcpy(local.data(),
                population + static_cast<size_t>(rank) * dimension,
                sizeof(double) * static_cast<size_t>(dimension));
    const CEDDetailedMetrics detailed = problem.detailed_metrics(local);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (rank == 0) {
      std::cout << "Algorithm = MadDE (standalone, NP=8)\n"
                << "Generation = " << completed_generations << '\n'
// 控制说明：选择当前编译配置对应的实现路径。
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
                << "Search exact evaluations = " << search_exact_evaluations
                << '\n'
#endif
                << "The best solution = " << best << '\n'
                << "Time = " << elapsed << " s\n";
    }
    sc::print_global_metrics(fitness[rank], detailed);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 0;
}
