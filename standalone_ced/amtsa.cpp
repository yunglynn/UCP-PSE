/* AMTSA paper-guided CED mapping: alternate structural search and
 * bottleneck-guided spatial/temporal splitting under a scalar exact objective. */
#include "benchmark_config.h"
#include "ced_problem.h"
#include "mpi_support.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>

namespace sc = standalone_ced;
namespace {
double clamp01(double x) { return std::max(0.0, std::min(1.0, x)); }

int tournament(const std::vector<double>& fitness, std::mt19937_64& rng) {
  int a = static_cast<int>(rng() % 8), b = static_cast<int>(rng() % 8);
  while (b == a) b = static_cast<int>(rng() % 8);
  return fitness[a] < fitness[b] ? a : b;
}

void route_structure_search(std::vector<double>& child, const double* peer,
                            const double* elite, const sc::CedProblem& p,
                            std::mt19937_64& rng) {
  const int t = p.task_count(), o = p.operation_count();
  const int mode = static_cast<int>(rng() % 3);
  if (mode == 0) {
    const int count = 1 + static_cast<int>(rng() % 16);
    for (int q = 0; q < count; ++q) {
      const int task = static_cast<int>(rng() % t);
      child[task] = peer[task];
      child[t + task] = peer[t + task];
    }
  } else if (mode == 1) {
    const int job = static_cast<int>(rng() % (o / p.operations_per_job()));
    const int first = 2 * t + job * p.operations_per_job();
    for (int q = 0; q < p.operations_per_job(); ++q) {
      child[first + q] = peer[first + q];
      child[2 * t + o + job * p.operations_per_job() + q] =
          peer[2 * t + o + job * p.operations_per_job() + q];
    }
  } else {
    const int start = static_cast<int>(rng() % o);
    const int length = std::min(o - start, 1 + static_cast<int>(rng() % 16));
    for (int q = 0; q < length; ++q) {
      const int j = 2 * t + start + q;
      child[j] = clamp01(0.5 * child[j] + 0.5 * elite[j]);
    }
    std::reverse(child.begin() + 2 * t + start,
                 child.begin() + 2 * t + start + length);
  }
}

void task_splitting_search(std::vector<double>& child, const double* elite,
                           const sc::CedProblem& p, std::mt19937_64& rng) {
  const int t = p.task_count(), o = p.operation_count();
  const int jobs = o / p.operations_per_job();
  int bottleneck_job = static_cast<int>(rng() % jobs);
  double largest = -1.0;
  for (int sample = 0; sample < std::min(16, jobs); ++sample) {
    const int job = static_cast<int>(rng() % jobs);
    double work = 0.0;
    for (int q = 0; q < p.operations_per_job(); ++q)
      work += p.operation_time(job * p.operations_per_job() + q);
    if (work > largest) { largest = work; bottleneck_job = job; }
  }
  const bool spatial = (rng() & 1U) == 0;
  if (spatial) {
    // Spatial splitting: redistribute the bottleneck job's operations among
    // capability-compatible devices through their random-key assignments.
    for (int q = 0; q < p.operations_per_job(); ++q) {
      const int operation = bottleneck_job * p.operations_per_job() + q;
      const int j = 2 * t + o + operation;
      const double offset = (static_cast<double>(q) + 0.5) /
                            p.operations_per_job();
      child[j] = clamp01(0.5 * elite[j] + 0.5 * offset);
    }
  } else {
    // Temporal splitting: spread the operation priorities of the bottleneck
    // job while retaining the elite ordering information.
    for (int q = 0; q < p.operations_per_job(); ++q) {
      const int operation = bottleneck_job * p.operations_per_job() + q;
      const int j = 2 * t + operation;
      const double target = (static_cast<double>(q) + 0.5) /
                            p.operations_per_job();
      child[j] = clamp01(0.5 * elite[j] + 0.5 * target);
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
                        static_cast<unsigned>(rank) * 67867967U);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<double> local(dimension), child(dimension);
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
      const int elite_index = static_cast<int>(
          std::min_element(fitness.begin(), fitness.end()) - fitness.begin());
      const int peer_index = tournament(fitness, rng);
      const double* elite = population.data() +
                            static_cast<size_t>(elite_index) * dimension;
      const double* peer = population.data() +
                           static_cast<size_t>(peer_index) * dimension;
      std::memcpy(child.data(), local.data(), sizeof(double) * dimension);
      const double progress = static_cast<double>(generation) /
                              std::max(1, MAXGEN - 1);
      if (uniform(rng) < 1.0 - progress)
        route_structure_search(child, peer, elite, problem, rng);
      else
        task_splitting_search(child, elite, problem, rng);
      const double child_fitness = problem.evaluate(child);
      if (child_fitness <= local_fitness) {
        local.swap(child);
        local_fitness = child_fitness;
      }
      std::memcpy(population.data() + static_cast<size_t>(rank) * dimension,
                  local.data(), sizeof(double) * dimension);
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
      std::cout << "Algorithm = AMTSA (paper-guided single-objective CED mapping, NP=8)\n"
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
