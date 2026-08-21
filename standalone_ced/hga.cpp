/* HGA CED adapter: genetic reproduction, hybrid local improvement, bounded
 * random-key decoding, exact evaluation, and survivor update. */
#include "benchmark_config.h"
#include "ced_problem.h"
#include "mpi_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace sc = standalone_ced;
namespace {
constexpr int kStagnationLimit = 20;
constexpr double kRankPressure = 1.8;

std::array<int, 4> block_begin(const sc::CedProblem& p) {
  const int t = p.task_count(), o = p.operation_count();
  return {0, t, 2 * t, 2 * t + o};
}
std::array<int, 4> block_end(const sc::CedProblem& p) {
  const int t = p.task_count(), o = p.operation_count();
  return {t, 2 * t, 2 * t + o, 2 * t + 2 * o};
}

int rank_select(const std::vector<double>& fitness, std::mt19937_64& rng) {
  std::array<int, sc::kPopulationSize> order{};
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return fitness[a] < fitness[b];
  });
  std::array<double, sc::kPopulationSize> weights{};
  for (int i = 0; i < sc::kPopulationSize; ++i)
    weights[i] = (2.0 - kRankPressure) +
                 2.0 * (kRankPressure - 1.0) * i /
                     (sc::kPopulationSize - 1.0);
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  return order[distribution(rng)];
}

void edd_initialization(std::vector<double>& x, const sc::CedProblem& p,
                        std::mt19937_64& rng) {
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (double& value : x) value = uniform(rng);
  const int task_priority = p.task_count();
  std::vector<int> tasks(p.task_count());
  std::iota(tasks.begin(), tasks.end(), 0);
  std::stable_sort(tasks.begin(), tasks.end(), [&](int a, int b) {
    return p.task_computation(a) + p.task_communication(a) <
           p.task_computation(b) + p.task_communication(b);
  });
  for (int position = 0; position < p.task_count(); ++position)
    x[task_priority + tasks[position]] =
        (position + uniform(rng)) / p.task_count();
  const int operation_priority = 2 * p.task_count() + p.operation_count();
  std::vector<int> operations(p.operation_count());
  std::iota(operations.begin(), operations.end(), 0);
  std::stable_sort(operations.begin(), operations.end(), [&](int a, int b) {
    return p.operation_time(a) < p.operation_time(b);
  });
  for (int position = 0; position < p.operation_count(); ++position)
    x[operation_priority + operations[position]] =
        (position + uniform(rng)) / p.operation_count();
}

void mutate(std::vector<double>& child, const sc::CedProblem& p,
            std::mt19937_64& rng) {
  const auto begin = block_begin(p), end = block_end(p);
  std::array<int, 4> blocks{0, 1, 2, 3};
  std::shuffle(blocks.begin(), blocks.end(), rng);
  const int block_count = 1 + static_cast<int>(rng() % 4);
  for (int q = 0; q < block_count; ++q) {
    const int lo = begin[blocks[q]], hi = end[blocks[q]];
    if (hi - lo < 2) continue;
    int a = lo + static_cast<int>(rng() % (hi - lo));
    int b = lo + static_cast<int>(rng() % (hi - lo));
    if (a > b) std::swap(a, b);
    if (a == b) b = std::min(hi - 1, a + 1);
    switch (rng() % 4) {
      case 0: std::shuffle(child.begin() + lo, child.begin() + hi, rng); break;
      case 1: std::swap(child[a], child[b]); break;
      case 2: std::reverse(child.begin() + a, child.begin() + b + 1); break;
      default: std::shuffle(child.begin() + a, child.begin() + b + 1, rng); break;
    }
  }
}
} // namespace

int main(int argc, char** argv) {
  try {
    sc::MpiSession mpi(&argc, &argv);
    sc::require_eight_processes(mpi.size());
    const int rank = mpi.rank();
    sc::CedProblem problem(sc::kCloudCount, sc::kEdgeCount, sc::kDeviceCount,
                           TNUM, MOPT_NUM, sc::kDataPath, sc::kPowerPath);
    const int dimension = problem.dimension();
    std::mt19937_64 rng(sc::experiment_seed() +
                        static_cast<unsigned>(rank) * 32452843U);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> local(dimension), child(dimension);
    for (double& value : local) value = uniform(rng);
    double local_fitness = problem.evaluate(local);
    std::vector<double> fitness;
    sc::gather_scalar(local_fitness, fitness);
    sc::trace_convergence(rank, 0, fitness);
    sc::SharedDoubleBuffer population(static_cast<size_t>(8) * dimension);
    sc::SharedDoubleBuffer offspring(static_cast<size_t>(8) * dimension);
    std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                local.data(), sizeof(double) * dimension);
    population.synchronize();
    double global_best = *std::min_element(fitness.begin(), fitness.end());
    int stagnant = 0;
    const double start = sc::wall_time();
    int completed_generations = 0;
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    const long long initial_exact_evaluations =
#if defined(INCLUDE_INITIAL_EXACT_EVALUATIONS) && \
    INCLUDE_INITIAL_EXACT_EVALUATIONS
        0;
#else
        sc::global_sum(problem.search_evaluation_count());
#endif
    while (completed_generations < MAXGEN &&
           sc::global_sum(problem.search_evaluation_count()) -
                   initial_exact_evaluations <
               SEARCH_EXACT_EVALUATION_BUDGET) {
      const int generation = completed_generations;
#else
    while (completed_generations < MAXGEN) {
      const int generation = completed_generations;
#endif
      int parent1 = rank_select(fitness, rng), parent2 = rank_select(fitness, rng);
      while (parent2 == parent1) parent2 = rank_select(fitness, rng);
      const double* p1 = population.data() + static_cast<size_t>(parent1) * dimension;
      const double* p2 = population.data() + static_cast<size_t>(parent2) * dimension;
      std::memcpy(child.data(), p1, sizeof(double) * dimension);
      const auto begin = block_begin(problem), end = block_end(problem);
      std::array<int, 4> blocks{0, 1, 2, 3};
      std::shuffle(blocks.begin(), blocks.end(), rng);
      const int chosen = 1 + static_cast<int>(rng() % 3);
      for (int q = 0; q < chosen; ++q) {
        const int lo = begin[blocks[q]], hi = end[blocks[q]];
        if (hi - lo < 2) continue;
        const int cut = lo + 1 + static_cast<int>(rng() % (hi - lo - 1));
        std::memcpy(child.data() + cut, p2 + cut,
                    sizeof(double) * (hi - cut));
      }
      if (rank == generation % sc::kPopulationSize) mutate(child, problem, rng);
      const double child_fitness = problem.evaluate(child);
      std::memcpy(offspring.data() + static_cast<size_t>(rank) * dimension,
                  child.data(), sizeof(double) * dimension);
      offspring.synchronize();
      std::vector<double> offspring_fitness;
      sc::gather_scalar(child_fitness, offspring_fitness);
      std::array<int, sc::kPopulationSize> selected{};
      if (rank == 0) {
        std::array<int, sc::kPopulationSize> parents{}, children{};
        std::iota(parents.begin(), parents.end(), 0);
        std::iota(children.begin(), children.end(), 0);
        std::sort(parents.begin(), parents.end(), [&](int a, int b) { return fitness[a] < fitness[b]; });
        std::sort(children.begin(), children.end(), [&](int a, int b) { return offspring_fitness[a] < offspring_fitness[b]; });
        for (int i = 0; i < 4; ++i) selected[i] = parents[i];
        for (int i = 0; i < 4; ++i) selected[4 + i] = 8 + children[i];
      }
#ifdef USE_MPI
      MPI_Bcast(selected.data(), 8, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      const int source = selected[rank];
      if (source < 8) {
        std::memcpy(local.data(), population.data() + static_cast<size_t>(source) * dimension,
                    sizeof(double) * dimension);
        local_fitness = fitness[source];
      } else {
        const int index = source - 8;
        std::memcpy(local.data(), offspring.data() + static_cast<size_t>(index) * dimension,
                    sizeof(double) * dimension);
        local_fitness = offspring_fitness[index];
      }
      sc::gather_scalar(local_fitness, fitness);
      const double current_best = *std::min_element(fitness.begin(), fitness.end());
      stagnant = current_best < global_best ? 0 : stagnant + 1;
      global_best = std::min(global_best, current_best);
      if (stagnant >= kStagnationLimit && rank >= 4) {
        edd_initialization(local, problem, rng);
        local_fitness = problem.evaluate(local);
      }
      if (stagnant >= kStagnationLimit) stagnant = 0;
      std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                  local.data(), sizeof(double) * dimension);
      population.synchronize();
      sc::gather_scalar(local_fitness, fitness);
      ++completed_generations;
      sc::trace_convergence(rank, completed_generations, fitness);
    }
    const double elapsed = sc::wall_time() - start;
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    const long long search_exact_evaluations =
        sc::global_sum(problem.search_evaluation_count()) -
        initial_exact_evaluations;
#endif
    const double best = *std::min_element(fitness.begin(), fitness.end());
    const CEDDetailedMetrics detailed = problem.detailed_metrics(local);
    if (rank == 0)
      std::cout << "Algorithm = HGA (standalone CED mapping, NP=8)\n"
                << "Generation = " << completed_generations << '\n'
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
    return 1;
  }
}
