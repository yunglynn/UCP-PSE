#define main ced_canonical_main_not_used
#include "main.cpp"
#undef main

#include "RecentSchedulingAlgorithms.h"

#include <memory>
#include <stdexcept>

namespace {
#ifndef ADE_TRUE_AUDIT_INTERVAL
#define ADE_TRUE_AUDIT_INTERVAL 0
#endif
enum class ComparisonMode {
  MemeRandom,
  ADERandom,
  MemePolicy,
  ADEPolicy,
  Recent
};

struct ComparisonChoice {
  ComparisonMode mode = ComparisonMode::MemePolicy;
  RecentAlgorithm recent = RecentAlgorithm::MadDE;
  std::string name = "meme_policy";
};

ComparisonChoice comparison_choice() {
  const char* value = std::getenv("CED_COMPARISON_ALG");
  const std::string name = value == nullptr ? "meme_policy" : value;
  ComparisonChoice choice;
  choice.name = name;
  if (name == "meme_random") choice.mode = ComparisonMode::MemeRandom;
  else if (name == "ade_random") choice.mode = ComparisonMode::ADERandom;
  else if (name == "meme_policy") choice.mode = ComparisonMode::MemePolicy;
  else if (name == "ade_policy") choice.mode = ComparisonMode::ADEPolicy;
  else if (ParseRecentAlgorithm(name, choice.recent))
    choice.mode = ComparisonMode::Recent;
  else
    throw std::runtime_error("Unknown CED_COMPARISON_ALG: " + name);
  return choice;
}

bool exact_recent_objective() {
  const char* value = std::getenv("CED_RECENT_EXACT_OBJECTIVE");
  return value != nullptr && std::atoi(value) != 0;
}

double proxy_reward(double old_fit, double new_fit, int true_cost,
                    clock_t work_cost) {
  if (!std::isfinite(old_fit) || !std::isfinite(new_fit) || new_fit >= old_fit)
    return 0.0;
  const double relative = (old_fit - new_fit) / (std::fabs(old_fit) + 1e-12);
  return std::sqrt(std::max(0.0, relative)) /
         std::sqrt(static_cast<double>(std::max<clock_t>(1, work_cost)) +
                   std::max(0, true_cost));
}

void update_proxy_weight(double& value, int trials, double reward) {
  const double rate = 1.0 / std::sqrt(static_cast<double>(std::max(1, trials)));
  value = reward > 0.0 ? value + rate * reward
                       : value * (1.0 - 0.25 * rate);
  value = std::max(0.04, std::min(3.5, value));
}

template <typename Search>
void with_proxy_policy(MultiMet& solver, int generation, int index,
                       Search&& search) {
  double progress = 1.0;
  const int state = solver.MemePolicyState(generation, MAXGEN, progress);
  const int action = solver.SelectPolicyProxyAction(state, progress);
  const double old_fit = solver.pop_fit[index];
  const int true_before = CED_ProxyTrueEvaluationCount();
  const clock_t work_start = clock();
  CED_SetProxyReduceTrueCheckHint(
      action == 0, solver.meme_policy.proxy[state][action]);
  search();
  CED_ClearProxyPolicyHint();
  const double reward = proxy_reward(
      old_fit, solver.newpop_fit[index],
      CED_ProxyTrueEvaluationCount() - true_before, clock() - work_start);
  int& trials = solver.proxy_policy_trials[state][action];
  ++trials;
  update_proxy_weight(solver.meme_policy.proxy[state][action], trials, reward);
}

void comparison_step(MultiMet& solver, RecentSchedulingSearch* recent,
                     const ComparisonChoice& choice, int generation,
                     int rank, int mpi_size, int p_start, int p_end,
                     unsigned base_seed) {
  unsigned generation_seed = base_seed + generation * 104729U;
#ifdef USE_MPI
  generation_seed += static_cast<unsigned>(rank) * 130363U;
#endif
  srand(generation_seed);

#ifdef USE_MPI
  if (!(choice.mode == ComparisonMode::Recent && exact_recent_objective()) &&
      choice.mode != ComparisonMode::ADERandom &&
      choice.mode != ComparisonMode::ADEPolicy)
    seed_local_working_slot_from_archive(solver, p_start, p_end);
#endif

  if (choice.mode == ComparisonMode::MemePolicy) {
    solver.RunMemeSearchStrategy(0, generation, MAXGEN,
                                 meme_search_scale(solver), p_start, p_end);
    solver.meme_state_update(p_start, p_end);
  } else if (choice.mode == ComparisonMode::MemeRandom) {
    for (int i = p_start; i < p_end; ++i)
      with_proxy_policy(solver, generation, i, [&] {
        solver.meme_selection(i, rand() % MultiMet::MemePolicyActionCount,
                              meme_search_scale(solver), 1);
      });
    solver.meme_state_update(p_start, p_end);
  } else if (choice.mode == ComparisonMode::ADERandom ||
             choice.mode == ComparisonMode::ADEPolicy) {
#if ADE_TRUE_AUDIT_INTERVAL > 0
    if (choice.mode == ComparisonMode::ADEPolicy &&
        (generation + 1) % ADE_TRUE_AUDIT_INTERVAL == 0) {
      const int audit_round =
          (generation + 1) / ADE_TRUE_AUDIT_INTERVAL - 1;
      if (rank == audit_round % std::max(1, mpi_size))
        CED_ForceNextProxyTrueEvaluation();
    }
#endif
    if (choice.mode == ComparisonMode::ADERandom)
      solver.ade_policy = MultiMet::default_seri_policy();
    for (int i = p_start; i < p_end; ++i)
      with_proxy_policy(solver, generation, i, [&] {
        solver.RunGlobalSearch(static_cast<int>(SearchAlg::ADE), generation,
                               MAXGEN, i, i + 1);
      });
    if (choice.mode == ComparisonMode::ADERandom)
      solver.ade_policy = MultiMet::default_seri_policy();
    solver.pop_better_update(p_start, p_end);
  } else {
    recent->Step(generation, MAXGEN, p_start, p_end);
  }

  solver.worst_and_best();
#ifdef USE_MPI
  if (sync_before_next_generation(generation, MIGRATION_INTERVAL))
    migrate_best_solution(solver, rank, mpi_size, p_start);
#else
  (void)mpi_size;
#endif
}
} // namespace

int main() {
#ifdef USE_MPI
  MPI_Init(nullptr, nullptr);
  int rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#else
  int rank = 0, mpi_size = 1;
#endif

  ComparisonChoice choice;
  try {
    choice = comparison_choice();
  } catch (const std::exception& error) {
    if (rank == 0) std::cerr << error.what() << std::endl;
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 2;
  }

  const int solver_population_size = TNUM >= 1000000 ? 1 : POPSIZE;
  int p_start = 0, p_end = solver_population_size;
#ifdef USE_MPI
  if (solver_population_size > 1)
    rank_range(rank, mpi_size, solver_population_size, p_start, p_end);
#endif

  const unsigned base_seed = make_seed(rank);
  srand(base_seed);
  FF comparison_objective =
      (choice.mode == ComparisonMode::Recent && exact_recent_objective())
          ? CED_Schedule
          : CED_Schedule_ParallelProxy;
  MultiMet solver(solver_population_size,
                  TNUM * 2 + TNUM * MOPT_NUM * 2, 0.0, 1.0,
                  CNUM, ENUM, DNUM, TNUM, TNUM, MOPT_NUM,
                  comparison_objective);
  solver.Initial();
#if defined(USE_LATEST_TASK_PRIOR_DESIGN) && USE_LATEST_TASK_PRIOR_DESIGN
  if (choice.mode == ComparisonMode::MemePolicy ||
      choice.mode == ComparisonMode::MemeRandom)
    apply_task_prior_proxy_design(solver);
#endif
#if defined(FIXED_STRUCTURED_INITIALIZATION) && FIXED_STRUCTURED_INITIALIZATION
  apply_task_priors(solver, rank);
#endif
#ifdef USE_MPI
  if (solver_population_size > 1)
    sync_initial_population(solver);
#endif
#ifdef CONVERGENCE_TRACE
  {
    double initial_verified = exact_recent_objective()
                                  ? solver.gbest_fit
                                  : CED_ProxyBestTrueValue();
#ifdef USE_MPI
    double global_initial_verified = initial_verified;
    MPI_Allreduce(&initial_verified, &global_initial_verified, 1, MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD);
    initial_verified = global_initial_verified;
#endif
    if (rank == 0)
      std::cout << "CONVERGENCE,0," << initial_verified << '\n';
  }
#endif
  std::unique_ptr<RecentSchedulingSearch> recent;
  if (choice.mode == ComparisonMode::Recent)
    recent.reset(new RecentSchedulingSearch(solver, choice.recent));

  const int prewarm_generations =
      comparison_objective == CED_Schedule_ParallelProxy
          ? proxy_prewarm_generations(solver.Popsize)
          : 0;
  for (int warmup = 0; warmup < prewarm_generations; ++warmup)
    comparison_step(solver, recent.get(), choice, warmup, rank, mpi_size,
                    p_start, p_end, base_seed);

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  const double start = current_time();
  int generation = 0;
  long long global_true_evaluations = 0;
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
  long long initial_true_evaluations = 0;
  {
    const long long local = CED_ProxyTrueEvaluationCount();
#ifdef USE_MPI
    MPI_Allreduce(&local, &global_true_evaluations, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
#else
    global_true_evaluations = local;
#endif
#if !defined(INCLUDE_INITIAL_EXACT_EVALUATIONS) || \
    !INCLUDE_INITIAL_EXACT_EVALUATIONS
    initial_true_evaluations = global_true_evaluations;
#endif
    global_true_evaluations -= initial_true_evaluations;
  }
#endif
#if defined(FIXED_GENERATION_BUDGET) && FIXED_GENERATION_BUDGET
  while (generation < MAXGEN) {
#elif defined(SEARCH_EXACT_EVALUATION_BUDGET)
  while (generation < MAXGEN &&
         global_true_evaluations < SEARCH_EXACT_EVALUATION_BUDGET) {
#else
  do {
#endif
    comparison_step(solver, recent.get(), choice, generation, rank, mpi_size,
                    p_start, p_end, base_seed);
    ++generation;
#ifdef CONVERGENCE_TRACE
    {
      double local_verified = exact_recent_objective()
                                  ? solver.gbest_fit
                                  : CED_ProxyBestTrueValue();
#ifdef USE_MPI
      double global_verified = local_verified;
      MPI_Allreduce(&local_verified, &global_verified, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
      local_verified = global_verified;
#endif
      if (rank == 0)
        std::cout << "CONVERGENCE," << generation << ',' << local_verified
                  << '\n';
    }
#endif
    long long local_count = CED_ProxyTrueEvaluationCount();
#ifdef USE_MPI
    long long total_count = 0;
    MPI_Allreduce(&local_count, &total_count, 1, MPI_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    global_true_evaluations = total_count - initial_true_evaluations;
#else
    global_true_evaluations = total_count;
#endif
#else
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    global_true_evaluations = local_count - initial_true_evaluations;
#else
    global_true_evaluations = local_count;
#endif
#endif
#if defined(FIXED_GENERATION_BUDGET) && FIXED_GENERATION_BUDGET
  }
#elif defined(SEARCH_EXACT_EVALUATION_BUDGET)
  }
#else
  } while (global_true_evaluations < 100);
#endif
  double elapsed = current_time() - start;
#ifdef USE_MPI
  double max_elapsed = elapsed;
  MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (rank == 0)
    elapsed = max_elapsed;
#endif
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
  const long long search_exact_evaluations = global_true_evaluations;
#endif

  std::vector<double> verified_candidate;
  double verified = verify_final_candidates_true(solver, p_start, p_end,
                                                  &verified_candidate);
  if (!verified_candidate.empty())
    true_schedule_value(solver, verified_candidate.data());
  CEDDetailedMetrics verified_metrics = CED_LastDetailedMetrics();
  long long local_proxy_actions[2] = {0, 0};
  for (int state = 0; state < MultiMet::HyperStateCount; ++state)
    for (int action = 0; action < MultiMet::MemeProxyPolicyActionCount;
         ++action)
      local_proxy_actions[action] += solver.proxy_policy_trials[state][action];
  long long global_proxy_actions[2] = {local_proxy_actions[0],
                                       local_proxy_actions[1]};
  long long local_true_evaluations = CED_ProxyTrueEvaluationCount();
  global_true_evaluations = local_true_evaluations;
  double local_ade_policy_deviation = 0.0;
  for (int state = 0; state < MultiMet::HyperStateCount; ++state) {
    for (int action = 0; action < MultiMet::AdeBaseModeCount; ++action)
      local_ade_policy_deviation +=
          std::fabs(solver.ade_policy.base[state][action] - 1.0);
    for (int action = 0; action < MultiMet::AdeMutationTypeCount; ++action)
      local_ade_policy_deviation +=
          std::fabs(solver.ade_policy.mutation[state][action] - 1.0);
    for (int action = 0; action < MultiMet::AdeMaxDiffTerms; ++action)
      local_ade_policy_deviation +=
          std::fabs(solver.ade_policy.diff[state][action] - 1.0);
    for (int action = 0; action < MultiMet::AdeDiffModeCount; ++action)
      local_ade_policy_deviation +=
          std::fabs(solver.ade_policy.diff_mode[state][action] - 1.0);
  }
  double global_ade_policy_deviation = local_ade_policy_deviation;
  double local_dtgp_alpha = recent ? recent->DTGPAlpha() : 0.0;
  double global_dtgp_alpha = local_dtgp_alpha;
  long long local_dtgp_nodes = recent ? recent->DTGPActiveNodes() : 0;
  long long global_dtgp_nodes = local_dtgp_nodes;
#ifdef USE_MPI
  double global_verified = verified;
  MPI_Allreduce(&verified, &global_verified, 1, MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD);
  verified = global_verified;
  MPI_Reduce(local_proxy_actions, global_proxy_actions, 2, MPI_LONG_LONG,
             MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_true_evaluations, &global_true_evaluations, 1,
             MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_ade_policy_deviation, &global_ade_policy_deviation, 1,
             MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_dtgp_alpha, &global_dtgp_alpha, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&local_dtgp_nodes, &global_dtgp_nodes, 1, MPI_LONG_LONG, MPI_SUM,
             0, MPI_COMM_WORLD);
#endif
  if (rank == 0) {
    std::cout << "Algorithm = " << choice.name << '\n';
    std::cout << "Objective mode = "
              << (exact_recent_objective() ? "exact" : "hybrid surrogate")
              << '\n';
    std::cout << "Generation = " << generation << '\n';
#ifdef SEARCH_EXACT_EVALUATION_BUDGET
    std::cout << "Search exact evaluations = " << search_exact_evaluations
              << '\n';
#endif
    std::cout << "The best solution = " << verified << '\n';
    std::cout << "Time = " << elapsed << " s\n";
    std::cout << "Proxy actions: reduce=" << global_proxy_actions[0]
              << ", conservative=" << global_proxy_actions[1]
              << ", true evaluations=" << global_true_evaluations << '\n';
    if (choice.mode == ComparisonMode::ADERandom ||
        choice.mode == ComparisonMode::ADEPolicy)
      std::cout << "ADE policy L1 deviation = "
                << global_ade_policy_deviation << '\n';
    if (choice.mode == ComparisonMode::Recent &&
        choice.recent == RecentAlgorithm::DTGPAM)
      std::cout << "DTGP alpha = " << recent->DTGPAlpha()
                << ", mean MPI alpha = "
                << global_dtgp_alpha / std::max(1, mpi_size)
                << ", active GP nodes = " << global_dtgp_nodes
                << '\n';
  }
  print_detailed_metrics(rank, verified_metrics.objective, verified_metrics);
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
