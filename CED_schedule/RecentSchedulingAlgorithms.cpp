/* Matched-budget baseline mechanisms. Each Step generates bounded CED random
 * keys, evaluates/selects, and updates method-specific state. NL-SHADE-LBC and
 * SLPSO-ARS are the two methods from this file used by the paper table. */
#include "RecentSchedulingAlgorithms.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <numeric>

namespace {
constexpr double kTiny = 1e-12;

double unit_random() { return (rand() + 0.5) / (RAND_MAX + 1.0); }

double clamp_value(double value, double low, double high) {
  return std::max(low, std::min(high, value));
}

// Deterministic numerical integration of the Beta(alpha,beta) density.  The
// DTGP-AM paper defines its three tree-selection probabilities as the masses
// over [0,.33], [.33,.66], and [.66,1].
double beta_interval_mass(double low, double high, double alpha, double beta) {
  constexpr int panels = 256;
  const double normalizer =
      std::tgamma(alpha + beta) / (std::tgamma(alpha) * std::tgamma(beta));
  const double width = (high - low) / panels;
  double integral = 0.0;
  for (int panel = 0; panel < panels; ++panel) {
    const double x = low + (panel + 0.5) * width;
    integral += normalizer * std::pow(x, alpha - 1.0) *
                std::pow(1.0 - x, beta - 1.0);
  }
  return integral * width;
}

void update_policy_weight(double& weight, int trials, double reward) {
  const double rate = 1.0 / std::sqrt(static_cast<double>(std::max(1, trials)));
  if (reward > 0.0)
    weight += rate * reward;
  else
    weight *= 1.0 - 0.25 * rate;
  weight = clamp_value(weight, 0.04, 3.5);
}
} // namespace

const char* RecentAlgorithmName(RecentAlgorithm algorithm) {
  switch (algorithm) {
  case RecentAlgorithm::MadDE: return "madde";
  case RecentAlgorithm::NLSHADELBC: return "nlshade_lbc";
  case RecentAlgorithm::SLPSOARS: return "slpso_ars";
  case RecentAlgorithm::DDEARA: return "dde_ara";
  case RecentAlgorithm::BiPopulationCDE: return "bipop_cde";
  case RecentAlgorithm::RLPHH: return "rl_phh";
  case RecentAlgorithm::DTGPAM: return "dtgp_am";
  case RecentAlgorithm::StandardDE: return "standard_de";
  case RecentAlgorithm::StandardPSO: return "standard_pso";
  case RecentAlgorithm::StandardGA: return "standard_ga";
  }
  return "unknown";
}

bool ParseRecentAlgorithm(const std::string& name, RecentAlgorithm& algorithm) {
  const std::array<RecentAlgorithm, 9> verified{{
      RecentAlgorithm::MadDE, RecentAlgorithm::NLSHADELBC,
      RecentAlgorithm::SLPSOARS, RecentAlgorithm::BiPopulationCDE,
      RecentAlgorithm::RLPHH, RecentAlgorithm::DTGPAM,
      RecentAlgorithm::StandardDE, RecentAlgorithm::StandardPSO,
      RecentAlgorithm::StandardGA}};
  for (RecentAlgorithm candidate : verified) {
    if (name == RecentAlgorithmName(candidate)) {
      algorithm = candidate;
      return true;
    }
  }
  return false;
}

RecentSchedulingSearch::RecentSchedulingSearch(MultiMet& solver,
                                               RecentAlgorithm algorithm)
    : solver_(solver), algorithm_(algorithm),
      region_radius_(solver.Popsize, (solver.Ubound - solver.Lbound) / 10.0) {
  shade_.f.fill(algorithm_ == RecentAlgorithm::NLSHADELBC ? 0.5 : 0.2);
  shade_.cr.fill(algorithm_ == RecentAlgorithm::NLSHADELBC ? 0.9 : 0.2);
  for (auto& state : rl_q_)
    state.fill(1.0);
  rl_state_ = rand() % static_cast<int>(rl_q_.size());
  for (auto& weights : gp_terminal_weight_)
    weights.fill(1.0 / GPTerminalCount);
  if (algorithm_ == RecentAlgorithm::DTGPAM) {
    gp_population_.resize(solver_.Popsize);
    for (int i = 0; i < solver_.Popsize; ++i) {
      const int depth = 2 + i % 5;
      const bool full = (i / 5) % 2 == 0;
      GenerateGPTree(gp_population_[i].vmsr, 0, 0, depth, full, 0, 0.0);
      GenerateGPTree(gp_population_[i].tsr, 0, 0, depth, full, 1, 0.0);
      gp_population_[i].fitness = solver_.pop_fit[i];
    }
  }
}

double RecentSchedulingSearch::Normal(double mean, double sd) const {
  const double u1 = std::max(kTiny, unit_random());
  const double u2 = unit_random();
  return mean + sd * std::sqrt(-2.0 * std::log(u1)) *
                    std::cos(2.0 * 3.14159265358979323846 * u2);
}

double RecentSchedulingSearch::Cauchy(double location, double scale) const {
  return location + scale *
                        std::tan(3.14159265358979323846 *
                                 (unit_random() - 0.5));
}

int RecentSchedulingSearch::DTGPActiveNodes() const {
  int count = 0;
  for (const GPIndividual& individual : gp_population_) {
    for (unsigned char node : individual.vmsr.node)
      count += node != 255;
    for (unsigned char node : individual.tsr.node)
      count += node != 255;
  }
  return count;
}

int RecentSchedulingSearch::DistinctIndex(int a, int b, int c) const {
  if (solver_.Popsize <= 1)
    return 0;
  for (int attempt = 0; attempt < 128; ++attempt) {
    const int value = rand() % solver_.Popsize;
    if (value != a && value != b && value != c)
      return value;
  }
  for (int value = 0; value < solver_.Popsize; ++value)
    if (value != a && value != b && value != c)
      return value;
  return a >= 0 ? a : 0;
}

std::vector<int> RecentSchedulingSearch::FitnessOrder() const {
  std::vector<int> order(solver_.Popsize);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return solver_.pop_fit[lhs] < solver_.pop_fit[rhs];
  });
  return order;
}

void RecentSchedulingSearch::Bound(double* candidate,
                                   const double* parent) const {
  for (int j = 0; j < solver_.Nvar; ++j) {
    if (candidate[j] < solver_.Lbound)
      candidate[j] = 0.5 * (solver_.Lbound + parent[j]);
    else if (candidate[j] > solver_.Ubound)
      candidate[j] = 0.5 * (solver_.Ubound + parent[j]);
  }
}

void RecentSchedulingSearch::EvaluateWithProxyPolicy(
    int index, int generation, int max_generation, double old_fit) {
  // The standalone exact-objective comparison constructs MultiMet with the
  // true CED objective.  In that protocol no proxy hint, proxy action, policy
  // update, or surrogate statistic may participate in selection.
  if (solver_.EvaluFunc != CED_Schedule_ParallelProxy) {
    solver_.Evaluation(true, index, index + 1);
    return;
  }
  double progress = 1.0;
  const int state = solver_.MemePolicyState(generation, max_generation,
                                             progress);
  const int action = solver_.SelectPolicyProxyAction(state, progress);
  const int true_before = CED_ProxyTrueEvaluationCount();
  const clock_t work_start = clock();
  CED_SetProxyReduceTrueCheckHint(
      action == 0, solver_.meme_policy.proxy[state][action]);
  solver_.Evaluation(true, index, index + 1);
  CED_ClearProxyPolicyHint();

  const int true_cost = CED_ProxyTrueEvaluationCount() - true_before;
  const double work_cost = static_cast<double>(
      std::max<clock_t>(1, clock() - work_start));
  double reward = 0.0;
  if (std::isfinite(old_fit) && std::isfinite(solver_.newpop_fit[index]) &&
      solver_.newpop_fit[index] < old_fit) {
    const double gain =
        (old_fit - solver_.newpop_fit[index]) / (std::fabs(old_fit) + kTiny);
    reward = std::sqrt(std::max(0.0, gain)) /
             std::sqrt(work_cost + std::max(0, true_cost));
  }
  int& trials = solver_.proxy_policy_trials[state][action];
  ++trials;
  update_policy_weight(solver_.meme_policy.proxy[state][action], trials,
                       reward);
}

void RecentSchedulingSearch::Step(int generation, int max_generation,
                                  int p_start, int p_end) {
  switch (algorithm_) {
  case RecentAlgorithm::MadDE:
    MadDEStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::NLSHADELBC:
    NLSHADELBStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::SLPSOARS:
    SLPSOARSStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::DDEARA:
    DDEARAStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::BiPopulationCDE:
    BiPopulationCDEStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::RLPHH:
    RLPHHStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::DTGPAM:
    DTGPAMStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::StandardDE:
    StandardDEStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::StandardPSO:
    StandardPSOStep(generation, max_generation, p_start, p_end); break;
  case RecentAlgorithm::StandardGA:
    StandardGAStep(generation, max_generation, p_start, p_end); break;
  }
  solver_.pop_better_update(p_start, p_end);
  solver_.worst_and_best();
  solver_.Elist();
}

void RecentSchedulingSearch::StandardDEStep(int generation,
                                            int max_generation,
                                            int p_start, int p_end) {
  constexpr double f = 0.5;
  constexpr double cr = 0.9;
  for (int i = p_start; i < p_end; ++i) {
    const int r1 = DistinctIndex(i);
    const int r2 = DistinctIndex(i, r1);
    const int r3 = DistinctIndex(i, r1, r2);
    const int forced = rand() % solver_.Nvar;
    for (int j = 0; j < solver_.Nvar; ++j) {
      const double mutant =
          solver_.pop[r1][j] + f * (solver_.pop[r2][j] - solver_.pop[r3][j]);
      solver_.newpop[i][j] =
          (j == forced || unit_random() <= cr) ? mutant : solver_.pop[i][j];
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    EvaluateWithProxyPolicy(i, generation, max_generation, solver_.pop_fit[i]);
  }
}

void RecentSchedulingSearch::StandardPSOStep(int generation,
                                             int max_generation,
                                             int p_start, int p_end) {
  constexpr double inertia = 0.729;
  constexpr double acceleration = 1.49445;
  const double velocity_limit = 0.2 * (solver_.Ubound - solver_.Lbound);
  if (solver_.velocity == nullptr) {
    solver_.velocity = solver_.CreateMatrix(solver_.Popsize, solver_.Nvar);
    for (int i = 0; i < solver_.Popsize; ++i)
      for (int j = 0; j < solver_.Nvar; ++j)
        solver_.velocity[i][j] =
            solver_.randval(-velocity_limit, velocity_limit);
  }
  for (int i = p_start; i < p_end; ++i) {
    for (int j = 0; j < solver_.Nvar; ++j) {
      const double r1 = unit_random();
      const double r2 = unit_random();
      double velocity =
          inertia * solver_.velocity[i][j] +
          acceleration * r1 * (solver_.ibest[i][j] - solver_.pop[i][j]) +
          acceleration * r2 * (solver_.gbest[j] - solver_.pop[i][j]);
      velocity = clamp_value(velocity, -velocity_limit, velocity_limit);
      solver_.velocity[i][j] = velocity;
      solver_.newpop[i][j] = solver_.pop[i][j] + velocity;
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    EvaluateWithProxyPolicy(i, generation, max_generation, solver_.pop_fit[i]);
  }
}

void RecentSchedulingSearch::StandardGAStep(int generation,
                                            int max_generation,
                                            int p_start, int p_end) {
  constexpr double crossover_probability = 0.9;
  constexpr double crossover_index = 20.0;
  constexpr double mutation_index = 20.0;

  double worst_fit = solver_.pop_fit[0];
  for (int i = 1; i < solver_.Popsize; ++i)
    worst_fit = std::max(worst_fit, solver_.pop_fit[i]);
  std::vector<double> roulette(solver_.Popsize, 0.0);
  double roulette_sum = 0.0;
  for (int i = 0; i < solver_.Popsize; ++i) {
    roulette[i] = worst_fit - solver_.pop_fit[i] + kTiny;
    roulette_sum += roulette[i];
  }
  auto select_parent = [&]() {
    if (!(roulette_sum > 0.0) || !std::isfinite(roulette_sum))
      return rand() % solver_.Popsize;
    double draw = unit_random() * roulette_sum;
    for (int i = 0; i < solver_.Popsize; ++i) {
      draw -= roulette[i];
      if (draw <= 0.0)
        return i;
    }
    return solver_.Popsize - 1;
  };

  const double mutation_probability = 1.0 / std::max(1, solver_.Nvar);
  for (int i = p_start; i < p_end; ++i) {
    const int first = select_parent();
    const int second = select_parent();
    for (int j = 0; j < solver_.Nvar; ++j) {
      const double x1 = solver_.pop[first][j];
      const double x2 = solver_.pop[second][j];
      double child = x1;
      if (unit_random() <= crossover_probability &&
          std::fabs(x1 - x2) > kTiny) {
        const double u = unit_random();
        const double beta = u <= 0.5
                                ? std::pow(2.0 * u,
                                           1.0 / (crossover_index + 1.0))
                                : std::pow(1.0 / (2.0 * (1.0 - u)),
                                           1.0 / (crossover_index + 1.0));
        const double sign = (rand() & 1) ? 1.0 : -1.0;
        child = 0.5 * ((1.0 + sign * beta) * x1 +
                       (1.0 - sign * beta) * x2);
      }
      if (unit_random() <= mutation_probability) {
        const double u = unit_random();
        const double delta =
            u < 0.5
                ? std::pow(2.0 * u, 1.0 / (mutation_index + 1.0)) - 1.0
                : 1.0 - std::pow(2.0 * (1.0 - u),
                                 1.0 / (mutation_index + 1.0));
        child += delta * (solver_.Ubound - solver_.Lbound);
      }
      solver_.newpop[i][j] = child;
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    EvaluateWithProxyPolicy(i, generation, max_generation, solver_.pop_fit[i]);
  }
}

void RecentSchedulingSearch::MadDEStep(int generation, int max_generation,
                                       int p_start, int p_end) {
  const std::vector<int> order = FitnessOrder();
  std::array<double, 3> improvement{{0.0, 0.0, 0.0}};
  const double progress = static_cast<double>(generation) /
                          std::max(1, max_generation);
  for (int i = p_start; i < p_end; ++i) {
    const int memory = rand() % static_cast<int>(shade_.f.size());
    double f;
    do f = Cauchy(shade_.f[memory], 0.1); while (f <= 0.0);
    f = std::min(1.0, f);
    const double cr = clamp_value(Normal(shade_.cr[memory], 0.1), 0.0, 1.0);
    const double draw = unit_random();
    int strategy = draw < madde_probability_[0]
                       ? 0
                       : (draw < madde_probability_[0] +
                                     madde_probability_[1] ? 1 : 2);
    const int r1 = DistinctIndex(i);
    const int r2 = DistinctIndex(i, r1);
    const int r3 = DistinctIndex(i, r1, r2);
    const int p_count = std::max(2, static_cast<int>(std::round(0.18 *
                                                               solver_.Popsize)));
    const int pbest = order[rand() % std::min(p_count, solver_.Popsize)];
    const int q_count = std::max(2, static_cast<int>(std::round(
                                      (0.36 - 0.18 * progress) * solver_.Popsize)));
    const int qbest = order[rand() % std::min(q_count, solver_.Popsize)];
    const bool use_qbest_crossover = unit_random() <= 0.01;
    const int forced = rand() % solver_.Nvar;
    for (int j = 0; j < solver_.Nvar; ++j) {
      double mutant;
      if (strategy == 0)
        mutant = solver_.pop[i][j] + f *
            (solver_.pop[pbest][j] - solver_.pop[i][j] +
             solver_.pop[r1][j] - solver_.ibest[r2][j]);
      else if (strategy == 1)
        mutant = solver_.pop[i][j] +
                 f * (solver_.pop[r1][j] - solver_.ibest[r2][j]);
      else
        mutant = f * (solver_.pop[r1][j] + (0.5 + 0.5 * progress) *
                      (solver_.pop[qbest][j] - solver_.pop[r3][j]));
      const double base = use_qbest_crossover ? solver_.pop[qbest][j]
                                               : solver_.pop[i][j];
      solver_.newpop[i][j] = (j == forced || unit_random() <= cr) ? mutant
                                                                  : base;
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    const double old_fit = solver_.pop_fit[i];
    EvaluateWithProxyPolicy(i, generation, max_generation, old_fit);
    if (solver_.newpop_fit[i] < old_fit) {
      improvement[strategy] += (old_fit - solver_.newpop_fit[i]) /
                               (std::fabs(old_fit) + kTiny);
      shade_.f[shade_.position] = f;
      shade_.cr[shade_.position] = cr;
      shade_.position = (shade_.position + 1) % shade_.f.size();
    }
  }
  const double total = improvement[0] + improvement[1] + improvement[2];
  if (total > 0.0)
    for (int s = 0; s < 3; ++s)
      madde_probability_[s] = clamp_value(improvement[s] / total, 0.1, 0.9);
  else
    madde_probability_ = {{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}};
  const double sum = madde_probability_[0] + madde_probability_[1] +
                     madde_probability_[2];
  for (double& probability : madde_probability_)
    probability /= sum;
}

void RecentSchedulingSearch::NLSHADELBStep(int generation, int max_generation,
                                           int p_start, int p_end) {
  const std::vector<int> order = FitnessOrder();
  const double progress = static_cast<double>(generation) /
                          std::max(1, max_generation);
  const int active = std::max(1, std::min(solver_.Popsize,
      static_cast<int>(std::round(
      (4.0 - solver_.Popsize) *
          std::pow(progress, std::max(0.0, 1.0 - progress)) +
      solver_.Popsize))));
  std::vector<int> fitness_rank(solver_.Popsize, 0);
  for (int rank = 0; rank < solver_.Popsize; ++rank)
    fitness_rank[order[rank]] = rank;
  std::vector<double> sorted_cr(active);
  for (double& value : sorted_cr) {
    const int memory = rand() % shade_.cr.size();
    value = clamp_value(Normal(shade_.cr[memory], 0.1), 0.0, 1.0);
  }
  std::sort(sorted_cr.begin(), sorted_cr.end());
  for (int i = p_start; i < p_end; ++i) {
    const int memory = rand() % shade_.f.size();
    double f;
    do f = Cauchy(shade_.f[memory], 0.1); while (f <= 0.0);
    f = std::min(1.0, f);
    const double cr = sorted_cr[std::min(active - 1, fitness_rank[i])];
    const int p_count = std::max(2, static_cast<int>(
        std::round(active * (0.2 + 0.1 * progress))));
    const int pbest = order[rand() % std::min(p_count, active)];
    const int r1 = order[rand() % active];
    int r2 = DistinctIndex(i, r1);
    const bool use_archive = unit_random() <= 0.5;
    if (!use_archive) {
      double total = 0.0;
      for (int rank = 0; rank < active; ++rank)
        total += std::exp(-static_cast<double>(rank) / active);
      double draw = unit_random() * total;
      for (int rank = 0; rank < active; ++rank) {
        draw -= std::exp(-static_cast<double>(rank) / active);
        if (draw <= 0.0) {
          r2 = order[rank];
          break;
        }
      }
    }
    const int forced = rand() % solver_.Nvar;
    for (int j = 0; j < solver_.Nvar; ++j) {
      const double mutant = solver_.pop[i][j] +
          f * (solver_.pop[pbest][j] - solver_.pop[i][j]) +
          f * (solver_.pop[r1][j] -
               (use_archive ? solver_.ibest[r2][j] : solver_.pop[r2][j]));
      solver_.newpop[i][j] =
          (j == forced || unit_random() <= cr) ? mutant : solver_.pop[i][j];
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    const double old_fit = solver_.pop_fit[i];
    EvaluateWithProxyPolicy(i, generation, max_generation, old_fit);
    if (solver_.newpop_fit[i] < old_fit) {
      // With one locally evaluated individual per MPI rank, the generalized
      // weighted Lehmer mean from the reference implementation reduces to
      // x^m, where m=1.5.  The reference then averages it with the old cell.
      shade_.f[shade_.position] =
          0.5 * (shade_.f[shade_.position] + std::pow(f, 1.5));
      shade_.cr[shade_.position] =
          0.5 * (shade_.cr[shade_.position] + std::pow(cr, 1.5));
      shade_.position = (shade_.position + 1) % shade_.f.size();
    } else {
      shade_.f[shade_.position] = 0.5;
      shade_.cr[shade_.position] = 0.5;
    }
  }
}

void RecentSchedulingSearch::SLPSOARSStep(int generation, int max_generation,
                                          int p_start, int p_end) {
  std::vector<int> worst_to_best = FitnessOrder();
  std::reverse(worst_to_best.begin(), worst_to_best.end());
  std::vector<int> rank(solver_.Popsize);
  for (int position = 0; position < solver_.Popsize; ++position)
    rank[worst_to_best[position]] = position;
  std::vector<double> mean(solver_.Nvar, 0.0);
  for (int p = 0; p < solver_.Popsize; ++p)
    for (int j = 0; j < solver_.Nvar; ++j)
      mean[j] += solver_.pop[p][j] / solver_.Popsize;
  const double exponent = 0.5 * std::log(std::ceil(solver_.Nvar / 100.0));
  for (int i = p_start; i < p_end; ++i) {
    const double probability = std::pow(
        std::max(0.0, 1.0 - static_cast<double>(rank[i]) / solver_.Popsize),
        exponent);
    std::memcpy(solver_.newpop[i], solver_.pop[i],
                sizeof(double) * solver_.Nvar);
    const bool moved = unit_random() <= probability &&
                       rank[i] + 1 < solver_.Popsize;
    if (moved) {
      for (int j = 0; j < solver_.Nvar; ++j) {
        const int guide_position = rank[i] + 1 +
            rand() % (solver_.Popsize - rank[i] - 1);
        const int guide = worst_to_best[guide_position];
        solver_.velocity[i][j] = unit_random() * solver_.velocity[i][j] +
            unit_random() * (solver_.pop[guide][j] - solver_.pop[i][j]) +
            0.1 * unit_random() * (mean[j] - solver_.pop[i][j]);
        solver_.newpop[i][j] += solver_.velocity[i][j];
      }
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    if (moved)
      EvaluateWithProxyPolicy(i, generation, max_generation,
                              solver_.pop_fit[i]);
    else
      solver_.newpop_fit[i] = solver_.pop_fit[i];
  }
  const double maximum_radius = (solver_.Ubound - solver_.Lbound) / 10.0 *
      (max_generation - generation + 1.0) / std::max(1, max_generation);
  const std::vector<int> best_order = FitnessOrder();
  for (int position = 0; position < std::min(5, solver_.Popsize); ++position) {
    const int i = best_order[position];
    if (i < p_start || i >= p_end)
      continue;
    bool success = false;
    for (int trial = 0; trial < 5; ++trial) {
      const int forced = rand() % solver_.Nvar;
      std::memcpy(solver_.newpop[i], solver_.pop[i],
                  sizeof(double) * solver_.Nvar);
      for (int j = 0; j < solver_.Nvar; ++j)
        if (j == forced || unit_random() < 0.01)
          solver_.newpop[i][j] += Normal(0.0, region_radius_[i]);
      Bound(solver_.newpop[i], solver_.pop[i]);
      EvaluateWithProxyPolicy(i, generation, max_generation,
                              solver_.pop_fit[i]);
      if (solver_.newpop_fit[i] < solver_.pop_fit[i]) {
        solver_.pop_better_update(i, i + 1);
        success = true;
      }
    }
    region_radius_[i] = success ? region_radius_[i] / 0.5
                                : region_radius_[i] * 0.5;
    region_radius_[i] = std::min(region_radius_[i], maximum_radius);
  }
}

void RecentSchedulingSearch::DDEARAStep(int generation, int max_generation,
                                        int p_start, int p_end) {
  // DDE-ARA allocates evaluation resources among heterogeneous DE islands.
  // Under the fixed one-island-per-rank CED layout, rank mod 3 identifies the
  // paper's constituent DE and improvement per evaluation is its GPI signal.
  for (int i = p_start; i < p_end; ++i) {
    const int family = i % 3;
    const int r1 = DistinctIndex(i);
    const int r2 = DistinctIndex(i, r1);
    const int r3 = DistinctIndex(i, r1, r2);
    const int forced = rand() % solver_.Nvar;
    for (int j = 0; j < solver_.Nvar; ++j) {
      double mutant;
      if (family == 0)
        mutant = solver_.pop[r1][j] + 0.5 *
                 (solver_.pop[r2][j] - solver_.pop[r3][j]);
      else if (family == 1)
        mutant = solver_.pop[i][j] + 0.5 *
                 (solver_.gbest[j] - solver_.pop[i][j]) + 0.5 *
                 (solver_.pop[r1][j] - solver_.pop[r2][j]);
      else
        mutant = solver_.gbest[j] + 0.5 *
                 (solver_.pop[r1][j] - solver_.pop[r2][j]);
      solver_.newpop[i][j] =
          (j == forced || unit_random() < 0.9) ? mutant : solver_.pop[i][j];
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    EvaluateWithProxyPolicy(i, generation, max_generation, solver_.pop_fit[i]);
  }
}

void RecentSchedulingSearch::BiPopulationCDEStep(
    int generation, int max_generation, int p_start, int p_end) {
  const std::vector<int> order = FitnessOrder();
  for (int i = p_start; i < p_end; ++i) {
    const bool local = i < solver_.Popsize / 2;
    const int r1 = DistinctIndex(i);
    const int r2 = DistinctIndex(i, r1);
    const int r3 = DistinctIndex(i, r1, r2);
    const int forced = rand() % solver_.Nvar;
    for (int j = 0; j < solver_.Nvar; ++j) {
      const double mutant = local
          ? solver_.pop[i][j] + 0.5 *
                (solver_.pop[order[0]][j] - solver_.pop[i][j]) +
                0.5 * (solver_.pop[r1][j] - solver_.pop[r2][j])
          : solver_.pop[r1][j] + 0.8 *
                (solver_.pop[r2][j] - solver_.pop[r3][j]);
      solver_.newpop[i][j] =
          (j == forced || unit_random() < 0.9) ? mutant : solver_.pop[i][j];
    }
    Bound(solver_.newpop[i], solver_.pop[i]);
    EvaluateWithProxyPolicy(i, generation, max_generation, solver_.pop_fit[i]);
  }
}

void RecentSchedulingSearch::RLPHHStep(int generation, int max_generation,
                                       int p_start, int p_end) {
  // Source-verified QPHH has three actions: precedence-safe swap, insertion,
  // and greedy insertion.  The random-key decoder preserves precedence, so
  // these actions map to priority-key swap, key insertion and best-parent
  // insertion while resource keys remain unchanged.
  const double epsilon = 0.1;
  for (int i = p_start; i < p_end; ++i) {
    int action = rand() % 3;
    if (unit_random() >= epsilon)
      action = static_cast<int>(std::max_element(rl_q_[rl_state_].begin(),
                       rl_q_[rl_state_].end()) - rl_q_[rl_state_].begin());
    std::memcpy(solver_.newpop[i], solver_.pop[i],
                sizeof(double) * solver_.Nvar);
    const int priority_offset = solver_.CE_Tnum;
    if (action == 0) {
      for (int operation = 0; operation < 3; ++operation) {
        const int a = rand() % solver_.CE_Tnum;
        const int b = rand() % solver_.CE_Tnum;
        std::swap(solver_.newpop[i][priority_offset + a],
                  solver_.newpop[i][priority_offset + b]);
      }
    } else if (action == 1) {
      for (int operation = 0; operation < 3; ++operation) {
        const int a = rand() % solver_.CE_Tnum;
        int b = rand() % solver_.CE_Tnum;
        if (operation == 1)
          b = rand() % std::max(1, solver_.CE_Tnum / 2);
        else if (operation == 2)
          b = solver_.CE_Tnum / 2 +
              rand() % std::max(1, solver_.CE_Tnum - solver_.CE_Tnum / 2);
        solver_.newpop[i][priority_offset + a] =
            0.5 * (solver_.newpop[i][priority_offset + a] +
                   solver_.newpop[i][priority_offset + b]);
      }
    } else {
      const int a = rand() % solver_.CE_Tnum;
      solver_.newpop[i][priority_offset + a] =
          solver_.gbest[priority_offset + a];
    }
    const double old_fit = solver_.pop_fit[i];
    EvaluateWithProxyPolicy(i, generation, max_generation, old_fit);
    const double log_improvement =
        std::log(std::max(kTiny, old_fit)) -
        std::log(std::max(kTiny, solver_.newpop_fit[i]));
    int next_state;
    if (log_improvement < 0.0) next_state = 0;
    else if (log_improvement == 0.0) next_state = 1;
    else if (log_improvement > 1.0) next_state = 11;
    else if (log_improvement > 1e-1) next_state = 2;
    else if (log_improvement > 1e-2) next_state = 3;
    else if (log_improvement > 1e-3) next_state = 4;
    else if (log_improvement > 1e-4) next_state = 5;
    else if (log_improvement > 1e-5) next_state = 6;
    else if (log_improvement > 1e-6) next_state = 7;
    else if (log_improvement > 1e-7) next_state = 8;
    else if (log_improvement > 1e-8) next_state = 9;
    else if (log_improvement > 1e-9) next_state = 10;
    else next_state = 12;
    double reward = 0.0;
    if (log_improvement < 0.0) reward = -10.0;
    else if (log_improvement >= 1e-1) reward = 10.0;
    else if (log_improvement >= 1e-2) reward = 9.0;
    else if (log_improvement >= 1e-3) reward = 8.0;
    else if (log_improvement >= 1e-4) reward = 7.0;
    else if (log_improvement >= 1e-5) reward = 6.0;
    else if (log_improvement >= 1e-6) reward = 5.0;
    const double alpha = 1.0 - 0.9 * generation /
                                   static_cast<double>(std::max(1, max_generation));
    const double future = *std::max_element(rl_q_[next_state].begin(),
                                             rl_q_[next_state].end());
    rl_q_[rl_state_][action] +=
        alpha * (reward + 0.5 * future - rl_q_[rl_state_][action]);
    rl_state_ = next_state;
  }
}

void RecentSchedulingSearch::GenerateGPTree(
    GPTree& tree, int index, int depth, int target_depth, bool full,
    int tree_type, double sampling_ratio) {
  if (index == 0 && depth == 0)
    tree.node.fill(255);
  if (index >= GPNodeCount)
    return;
  const bool terminal = depth >= target_depth ||
      (!full && depth >= 2 && unit_random() < 0.35);
  if (terminal) {
    const int terminal_count = tree_type == 0 ? GPTerminalCount : 7;
    int selected = rand() % terminal_count;
    if (sampling_ratio > 0.0 && unit_random() < sampling_ratio) {
      double draw = unit_random();
      selected = terminal_count - 1;
      for (int terminal_index = 0; terminal_index < terminal_count;
           ++terminal_index) {
        draw -= gp_terminal_weight_[tree_type][terminal_index];
        if (draw <= 0.0) {
          selected = terminal_index;
          break;
        }
      }
    }
    tree.node[index] = static_cast<unsigned char>(6 + selected);
    return;
  }
  tree.node[index] = static_cast<unsigned char>(rand() % 6);
  GenerateGPTree(tree, 2 * index + 1, depth + 1, target_depth, full,
                 tree_type, sampling_ratio);
  GenerateGPTree(tree, 2 * index + 2, depth + 1, target_depth, full,
                 tree_type, sampling_ratio);
}

double RecentSchedulingSearch::EvaluateGPTree(
    const GPTree& tree, int index,
    const std::array<double, GPTerminalCount>& terminals) const {
  if (index >= GPNodeCount || tree.node[index] == 255)
    return 0.0;
  const int kind = tree.node[index];
  if (kind >= 6)
    return terminals[std::min(GPTerminalCount - 1, kind - 6)];
  const double left = EvaluateGPTree(tree, 2 * index + 1, terminals);
  const double right = EvaluateGPTree(tree, 2 * index + 2, terminals);
  double value = 0.0;
  switch (kind) {
  case 0: value = left + right; break;
  case 1: value = left - right; break;
  case 2: value = left * right; break;
  case 3: value = std::fabs(right) > kTiny ? left / right : left; break;
  case 4: value = std::max(left, right); break;
  default: value = std::min(left, right); break;
  }
  return clamp_value(value, -1e6, 1e6);
}

void RecentSchedulingSearch::SwapGPSubtree(GPTree& first, GPTree& second,
                                            int root) {
  if (root >= GPNodeCount)
    return;
  std::swap(first.node[root], second.node[root]);
  SwapGPSubtree(first, second, 2 * root + 1);
  SwapGPSubtree(first, second, 2 * root + 2);
}

void RecentSchedulingSearch::MutateGPSubtree(GPTree& tree, int tree_type,
                                              double sampling_ratio) {
  std::vector<int> active;
  for (int index = 0; index < GPNodeCount; ++index)
    if (tree.node[index] != 255)
      active.push_back(index);
  const int root = active.empty() ? 0 : active[rand() % active.size()];
  std::vector<int> stack(1, root);
  while (!stack.empty()) {
    const int index = stack.back();
    stack.pop_back();
    if (index >= GPNodeCount)
      continue;
    tree.node[index] = 255;
    stack.push_back(2 * index + 1);
    stack.push_back(2 * index + 2);
  }
  int root_depth = 0;
  for (int value = root + 1; value > 1; value >>= 1)
    ++root_depth;
  const int target = root_depth + 1 + rand() %
      std::max(1, GPMaxDepth - root_depth);
  GenerateGPTree(tree, root, root_depth, target, unit_random() < 0.5,
                 tree_type, sampling_ratio);
}

int RecentSchedulingSearch::GPTournament() const {
  int winner = rand() % solver_.Popsize;
  for (int draw = 1; draw < 7; ++draw) {
    const int candidate = rand() % solver_.Popsize;
    if (gp_population_[candidate].fitness < gp_population_[winner].fitness)
      winner = candidate;
  }
  return winner;
}

void RecentSchedulingSearch::UpdateGPTerminalWeights(double sampling_ratio) {
  std::vector<int> order(solver_.Popsize);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
    return gp_population_[lhs].fitness < gp_population_[rhs].fitness;
  });
  const int selected_count = std::max(1, solver_.Popsize / 2);
  const double r = std::max(1e-6, sampling_ratio);
  for (int tree_type = 0; tree_type < 2; ++tree_type) {
    const int terminal_count = tree_type == 0 ? GPTerminalCount : 7;
    const double offset = (1.0 - r) / (terminal_count * r);
    std::array<double, GPTerminalCount> count{};
    for (int rank = 0; rank < selected_count; ++rank) {
      const GPTree& tree = tree_type == 0
          ? gp_population_[order[rank]].vmsr
          : gp_population_[order[rank]].tsr;
      for (unsigned char kind : tree.node)
        if (kind >= 6 && kind < 6 + terminal_count)
          count[kind - 6] += 1.0;
    }
    const double total = std::accumulate(count.begin(), count.end(), 0.0);
    if (total <= 0.0) {
      gp_terminal_weight_[tree_type].fill(0.0);
      for (int terminal = 0; terminal < terminal_count; ++terminal)
        gp_terminal_weight_[tree_type][terminal] = 1.0 / terminal_count;
      continue;
    }
    auto mass = [&](double nu) {
      double sum = 0.0;
      for (int terminal = 0; terminal < terminal_count; ++terminal)
        sum += std::max(0.0, count[terminal] / nu - offset);
      return sum;
    };
    double low = 1e-12, high = total;
    while (mass(high) > 1.0)
      high *= 2.0;
    for (int iteration = 0; iteration < 80; ++iteration) {
      const double middle = 0.5 * (low + high);
      if (mass(middle) > 1.0)
        low = middle;
      else
        high = middle;
    }
    double sum = 0.0;
    for (int terminal = 0; terminal < terminal_count; ++terminal) {
      gp_terminal_weight_[tree_type][terminal] =
          std::max(0.0, count[terminal] / high - offset);
      sum += gp_terminal_weight_[tree_type][terminal];
    }
    if (sum <= 0.0) {
      gp_terminal_weight_[tree_type].fill(0.0);
      for (int terminal = 0; terminal < terminal_count; ++terminal)
        gp_terminal_weight_[tree_type][terminal] = 1.0 / terminal_count;
    } else {
      for (double& weight : gp_terminal_weight_[tree_type])
        weight /= sum;
    }
  }
}

void RecentSchedulingSearch::DTGPAMStep(int generation, int max_generation,
                                        int p_start, int p_end) {
  for (int i = 0; i < solver_.Popsize; ++i)
    gp_population_[i].fitness = solver_.pop_fit[i];
  const double sampling_ratio = generation <= 0 ? 0.0 :
      1.0 - 1.0 / (1.0 + 2.0 * std::log(generation + 1.0));
  UpdateGPTerminalWeights(sampling_ratio);
  const double p_both = beta_interval_mass(0.0, 0.33, gp_alpha_, 8.0);
  const double p_resource = beta_interval_mass(0.33, 0.66, gp_alpha_, 8.0);
  const double p_task = beta_interval_mass(0.66, 1.0, gp_alpha_, 8.0);
  const double sum = p_both + p_resource + p_task;
  int mutation_trials = 0, mutation_successes = 0;
  for (int i = p_start; i < p_end; ++i) {
    const int parent = GPTournament();
    GPIndividual offspring = gp_population_[parent];
    bool mutation = false;
    const double operator_draw = unit_random();
    if (operator_draw < 0.65) {
      const int second = GPTournament();
      const bool resource_tree = unit_random() < 0.5;
      GPTree& first_tree = resource_tree ? offspring.vmsr : offspring.tsr;
      GPTree second_tree = resource_tree ? gp_population_[second].vmsr
                                         : gp_population_[second].tsr;
      std::vector<int> active;
      for (int node = 0; node < GPNodeCount; ++node)
        if (first_tree.node[node] != 255 && second_tree.node[node] != 255)
          active.push_back(node);
      SwapGPSubtree(first_tree, second_tree,
                    active.empty() ? 0 : active[rand() % active.size()]);
    } else if (operator_draw < 0.95) {
      mutation = true;
      ++mutation_trials;
      const double draw = unit_random() * sum;
      if (draw < p_both || draw < p_both + p_resource)
        MutateGPSubtree(offspring.vmsr, 0, sampling_ratio);
      if (draw < p_both || draw >= p_both + p_resource)
        MutateGPSubtree(offspring.tsr, 1, sampling_ratio);
    }
    const double progress = static_cast<double>(generation + 1) /
                            std::max(1, max_generation);
    auto logistic = [](double value) {
      value = clamp_value(value, -30.0, 30.0);
      return 1.0 / (1.0 + std::exp(-value));
    };
    for (int task = 0; task < solver_.CE_Tnum; ++task) {
      const double comp = solver_.CETask_Property[task].Computation / 199.0;
      const double comm = solver_.CETask_Property[task].Communication / 4999.0;
      const double relation =
          (solver_.CETask_Property[task].Precedence.size() +
           solver_.CETask_Property[task].Interact.size() +
           solver_.CETask_Property[task].Start_Pre.size() +
           solver_.CETask_Property[task].End_Pre.size()) / 8.0;
      const double choices =
          solver_.CETask_Property[task].AvailEdgeServerList.size() / 4.0;
      const double position = static_cast<double>(task) /
                              std::max(1, solver_.CE_Tnum - 1);
      std::array<double, GPTerminalCount> terminals{{
          comp, comm, choices, 1.0 - progress, relation, position,
          comp + comm, comp - comm, relation + choices, progress, 1.0}};
      const double score = logistic(EvaluateGPTree(offspring.vmsr, 0,
                                                    terminals));
      solver_.newpop[i][task] = score;
      solver_.newpop[i][solver_.CE_Tnum + task] =
          std::fmod(score + 0.61803398875, 1.0);
    }
    const int operations = solver_.M_Jnum * solver_.M_OPTnum;
    const int sequence_base = 2 * solver_.CE_Tnum;
    const int device_base = sequence_base + operations;
    for (int operation = 0; operation < operations; ++operation) {
      const double duration = solver_.MTask_Time[operation] / 300.0;
      const double choices = solver_.AvailDeviceList[operation].size() / 3.0;
      const double position = static_cast<double>(operation) /
                              std::max(1, operations - 1);
      std::array<double, GPTerminalCount> task_terminals{{
          duration, 1.0 - progress, position, choices,
          duration + position, progress, 1.0, 0.0, 0.0, 0.0, 0.0}};
      solver_.newpop[i][sequence_base + operation] = logistic(
          EvaluateGPTree(offspring.tsr, 0, task_terminals));
      std::array<double, GPTerminalCount> device_terminals{{
          duration, choices, position, progress, 1.0 - progress,
          duration + choices, duration - choices, position,
          choices + position, 1.0, 0.5}};
      solver_.newpop[i][device_base + operation] = logistic(
          EvaluateGPTree(offspring.vmsr, 0, device_terminals));
    }
    const double old_fit = solver_.pop_fit[i];
    EvaluateWithProxyPolicy(i, generation, max_generation, old_fit);
    offspring.fitness = solver_.newpop_fit[i];
    gp_population_[i] = offspring;
    if (mutation && offspring.fitness < old_fit)
      ++mutation_successes;
    const size_t bytes = sizeof(double) * static_cast<size_t>(solver_.Nvar);
    std::memcpy(solver_.pop[i], solver_.newpop[i], bytes);
    solver_.pop_fit[i] = solver_.newpop_fit[i];
    if (solver_.newpop_fit[i] < solver_.ibest_fit[i]) {
      std::memcpy(solver_.ibest[i], solver_.newpop[i], bytes);
      solver_.ibest_fit[i] = solver_.newpop_fit[i];
    }
    if (solver_.newpop_fit[i] < solver_.gbest_fit) {
      std::memcpy(solver_.gbest, solver_.newpop[i], bytes);
      solver_.gbest_fit = solver_.newpop_fit[i];
    }
  }
  if (mutation_trials > 0) {
    const double current = static_cast<double>(mutation_successes) /
                           mutation_trials;
    gp_smoothed_success_ +=
        (2.0 / 11.0) * (current - gp_smoothed_success_);
    if (gp_smoothed_success_ < 0.2)
      gp_alpha_ /= 0.817;
    else if (gp_smoothed_success_ > 0.2)
      gp_alpha_ *= 0.817;
    gp_alpha_ = clamp_value(gp_alpha_, 2.0, 23.0);
  }
}
