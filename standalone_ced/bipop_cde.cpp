/* Bi-Population CDE mapping: decision-block local/global roles,
 * current-to-best/1 and rand/1, greedy selection, diversity regeneration. */
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
constexpr double kF = 0.5;
constexpr double kCR = 0.7;
constexpr double kDiversityThreshold = 0.1;

std::array<int, 5> boundaries(const sc::CedProblem& p) {
  const int t = p.task_count(), o = p.operation_count();
  return {0, t, 2 * t, 2 * t + o, 2 * t + 2 * o};
}
double reflect(double value) {
  value = std::fmod(value, 2.0);
  if (value < 0.0) value += 2.0;
  return value <= 1.0 ? value : 2.0 - value;
}
int distinct_index(const std::array<int, 4>& pool, int avoid1, int avoid2,
                   int avoid3,
                   std::mt19937_64& rng) {
  int value;
  do value = pool[rng() % pool.size()];
  while (value == avoid1 || value == avoid2 || value == avoid3);
  return value;
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
    const auto block = boundaries(problem);
    std::mt19937_64 rng(sc::experiment_seed() +
                        static_cast<unsigned>(rank) * 49979687U);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> local(dimension), trial(dimension);
    for (double& value : local) value = uniform(rng);
    double local_fitness = problem.evaluate(local);
    std::vector<double> fitness;
    sc::gather_scalar(local_fitness, fitness);
    sc::trace_convergence(rank, 0, fitness);
    sc::SharedDoubleBuffer population(static_cast<size_t>(8) * dimension);
    std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                local.data(), sizeof(double) * dimension);
    population.synchronize();
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
      std::array<int, 8> order{};
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](int a, int b) {
        return fitness[a] < fitness[b];
      });
      std::array<int, 4> local_set{}, global_set{};
      std::copy_n(order.begin(), 4, local_set.begin());
      std::copy_n(order.begin() + 4, 4, global_set.begin());
      const bool is_local = std::find(local_set.begin(), local_set.end(), rank) != local_set.end();
      const int active = generation % 4;
      const int lo = block[active], hi = block[active + 1];
      std::memcpy(trial.data(), local.data(), sizeof(double) * dimension);
      int r1, r2, r3 = -1;
      if (is_local) {
        r1 = distinct_index(local_set, rank, -1, -1, rng);
        r2 = distinct_index(local_set, rank, r1, -1, rng);
      } else {
        r1 = distinct_index(global_set, rank, -1, -1, rng);
        r2 = distinct_index(global_set, rank, r1, -1, rng);
        r3 = distinct_index(global_set, rank, r1, r2, rng);
      }
      const int forced = lo + static_cast<int>(rng() % (hi - lo));
      for (int j = lo; j < hi; ++j) {
        if (uniform(rng) > kCR && j != forced) continue;
        double mutant;
        if (is_local) {
          const double best = population.data()[static_cast<size_t>(order[0]) * dimension + j];
          const double a = population.data()[static_cast<size_t>(r1) * dimension + j];
          const double b = population.data()[static_cast<size_t>(r2) * dimension + j];
          mutant = local[j] + kF * (best - local[j]) + kF * (a - b);
        } else {
          const double a = population.data()[static_cast<size_t>(r1) * dimension + j];
          const double b = population.data()[static_cast<size_t>(r2) * dimension + j];
          const double c = population.data()[static_cast<size_t>(r3) * dimension + j];
          mutant = a + kF * (b - c);
        }
        trial[j] = reflect(mutant);
      }
      const double trial_fitness = problem.evaluate(trial);
      if (trial_fitness <= local_fitness) {
        local.swap(trial);
        local_fitness = trial_fitness;
      }
      std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                  local.data(), sizeof(double) * dimension);
      population.synchronize();
      sc::gather_scalar(local_fitness, fitness);

      double diversity = 0.0;
      if (!is_local) {
        const int samples = std::min(64, hi - lo);
        for (int s = 0; s < samples; ++s) {
          const int j = lo + static_cast<int>((static_cast<long long>(s) * (hi - lo)) / samples);
          double mean = 0.0;
          for (int index : global_set)
            mean += population.data()[static_cast<size_t>(index) * dimension + j] / 4.0;
          for (int index : global_set) {
            const double delta = population.data()[static_cast<size_t>(index) * dimension + j] - mean;
            diversity += delta * delta;
          }
        }
        diversity = std::sqrt(diversity / std::max(1, 4 * samples));
      }
      std::vector<double> diversities;
      sc::gather_scalar(diversity, diversities);
      double global_diversity = 0.0;
      for (int index : global_set) global_diversity = std::max(global_diversity, diversities[index]);
      if (!is_local && global_diversity < kDiversityThreshold) {
        for (int j = lo; j < hi; ++j) local[j] = uniform(rng);
        local_fitness = problem.evaluate(local);
        std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                    local.data(), sizeof(double) * dimension);
      }
      population.synchronize();
      sc::gather_scalar(local_fitness, fitness);
      ++completed_generations;
      sc::trace_convergence(rank, completed_generations, fitness);
    }
    const double elapsed = sc::wall_time() - start;
    const CEDDetailedMetrics detailed = problem.detailed_metrics(local);
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    const long long search_exact_evaluations =
        sc::global_sum(problem.search_evaluation_count()) -
        initial_exact_evaluations;
#endif
    if (rank == 0)
      std::cout << "Algorithm = Bi-Population CDE (paper-guided CED mapping, NP=8)\n"
                << "Generation = " << completed_generations << '\n'
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
                << "Search exact evaluations = "
                << search_exact_evaluations
                << '\n'
#endif
                << "The best solution = "
                << *std::min_element(fitness.begin(), fitness.end()) << '\n'
                << "Time = " << elapsed << " s\n";
    sc::print_global_metrics(local_fitness, detailed);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
