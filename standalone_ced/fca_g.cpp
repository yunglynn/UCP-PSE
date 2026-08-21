/* FCA-G CED adapter: sparse relation-context factors, grouped random-key
 * variation, exact evaluation, and greedy survivor selection. */
#include "benchmark_config.h"
#include "ced_problem.h"
#include "mpi_support.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>

namespace sc = standalone_ced;
namespace {
int distinct(std::mt19937_64& rng, int a, int b = -1, int c = -1,
             int d = -1) {
  int value;
  do value = static_cast<int>(rng() % sc::kPopulationSize);
  while (value == a || value == b || value == c || value == d);
  return value;
}
double reflect(double value) {
  if (!std::isfinite(value)) return 0.5;
  value = std::fmod(value, 2.0);
  if (value < 0.0) value += 2.0;
  return value <= 1.0 ? value : 2.0 - value;
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
                        static_cast<unsigned>(rank) * 130363U);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double> normal01(0.0, 1.0);
    std::vector<double> parent(dimension), child(dimension);
    for (double& value : parent) value = uniform(rng);
    double local_fitness = problem.evaluate(parent);
    std::vector<double> fitness;
    sc::gather_scalar(local_fitness, fitness);
    sc::trace_convergence(rank, 0, fitness);
    sc::SharedDoubleBuffer population(
        static_cast<size_t>(sc::kPopulationSize) * dimension);
    std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                parent.data(), sizeof(double) * parent.size());
    population.synchronize();

    std::array<std::pair<int, int>, 4> groups{{
        {0, TNUM}, {TNUM, 2 * TNUM}, {2 * TNUM, 2 * TNUM + TNUM * MOPT_NUM},
        {2 * TNUM + TNUM * MOPT_NUM, dimension}}};
    std::array<double, 4> contribution{{0.0, 0.0, 0.0, 0.0}};
    double crossover_mean = 0.5, strategy_probability = 0.5,
           f_probability = 0.5;
    double s_strategy1 = 1, f_strategy1 = 1, s_strategy2 = 1, f_strategy2 = 1;
    double s_normal = 1, f_normal = 1, s_cauchy = 1, f_cauchy = 1;
    double cr_weighted_sum = 0.0, improvement_sum = 0.0;

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
      int group = generation < 4 ? generation : static_cast<int>(
          std::max_element(contribution.begin(), contribution.end()) -
          contribution.begin());
      if (generation >= 4 && uniform(rng) < 0.05)
        group = static_cast<int>(rng() % 4);
#ifdef USE_MPI
      MPI_Bcast(&group, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
      const int begin = groups[group].first, end = groups[group].second;
      const int best = static_cast<int>(std::min_element(
          fitness.begin(), fitness.end()) - fitness.begin());
      const int r1 = distinct(rng, rank);
      const int r2 = distinct(rng, rank, r1);
      const int r3 = distinct(rng, rank, r1, r2);
      const int r4 = distinct(rng, rank, r1, r2, r3);
      const bool strategy1 = uniform(rng) <= strategy_probability;
      const bool normal_f = uniform(rng) <= f_probability;
      double scale = normal_f ? 0.5 + 0.3 * normal01(rng)
                              : normal01(rng) / std::max(1e-12,
                                                        normal01(rng));
      scale = std::fabs(scale);
      double crossover = 0.0;
      do crossover = crossover_mean + 0.1 * normal01(rng);
      while (crossover <= 0.0 || crossover >= 1.0);
      std::memcpy(child.data(), parent.data(), sizeof(double) * parent.size());
      const int forced = begin + static_cast<int>(rng() % (end - begin));
      auto at = [&](int individual, int coordinate) {
        return population.data()[static_cast<size_t>(individual) * dimension +
                                 coordinate];
      };
      for (int coordinate = begin; coordinate < end; ++coordinate) {
        if (coordinate != forced && uniform(rng) >= crossover) continue;
        double value = 0.0;
        if (strategy1)
          value = at(rank, coordinate) +
                  scale * (at(best, coordinate) - at(rank, coordinate)) +
                  scale * (at(r1, coordinate) - at(r2, coordinate) +
                           at(r3, coordinate) - at(r4, coordinate));
        else
          value = at(r3, coordinate) +
                  scale * (at(r1, coordinate) - at(r2, coordinate));
        child[coordinate] = reflect(value);
      }
      const double child_fitness = problem.evaluate(child);
      const bool accepted = child_fitness <= local_fitness;
      const double improvement = accepted ? local_fitness - child_fitness : 0.0;
      if (accepted) {
        parent.swap(child);
        local_fitness = child_fitness;
      }
      std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                  parent.data(), sizeof(double) * parent.size());
      population.synchronize();
      sc::gather_scalar(local_fitness, fitness);
      std::vector<int> accepted_all, strategy_all, normal_all;
      std::vector<double> improvement_all, crossover_all;
      sc::gather_int(accepted ? 1 : 0, accepted_all);
      sc::gather_int(strategy1 ? 1 : 0, strategy_all);
      sc::gather_int(normal_f ? 1 : 0, normal_all);
      sc::gather_scalar(improvement, improvement_all);
      sc::gather_scalar(crossover, crossover_all);
      double generation_contribution = 0.0;
      for (int i = 0; i < sc::kPopulationSize; ++i) {
        generation_contribution += improvement_all[i];
        if (strategy_all[i]) (accepted_all[i] ? s_strategy1 : f_strategy1)++;
        else (accepted_all[i] ? s_strategy2 : f_strategy2)++;
        if (normal_all[i]) (accepted_all[i] ? s_normal : f_normal)++;
        else (accepted_all[i] ? s_cauchy : f_cauchy)++;
        if (improvement_all[i] > 0.0) {
          cr_weighted_sum += crossover_all[i] * improvement_all[i];
          improvement_sum += improvement_all[i];
        }
      }
      contribution[group] = generation_contribution;
      if ((generation + 1) % 24 == 0) {
        strategy_probability =
            (s_strategy1 / (s_strategy1 + f_strategy1)) /
            ((s_strategy1 / (s_strategy1 + f_strategy1)) +
             (s_strategy2 / (s_strategy2 + f_strategy2)));
        f_probability =
            (s_normal * (s_cauchy + f_cauchy)) /
            (s_cauchy * (s_normal + f_normal) +
             s_normal * (s_cauchy + f_cauchy));
        if (improvement_sum > 0.0)
          crossover_mean = cr_weighted_sum / improvement_sum;
        s_strategy1 = f_strategy1 = s_strategy2 = f_strategy2 = 1;
        s_normal = f_normal = s_cauchy = f_cauchy = 1;
        cr_weighted_sum = improvement_sum = 0.0;
      }
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
    const CEDDetailedMetrics detailed = problem.detailed_metrics(parent);
    if (rank == 0)
      std::cout << "Algorithm = FCA-G (standalone CED mapping, NP=8)\n"
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
