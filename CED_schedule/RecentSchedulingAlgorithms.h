#ifndef CED_SCHEDULE_RECENT_SCHEDULING_ALGORITHMS_H
#define CED_SCHEDULE_RECENT_SCHEDULING_ALGORITHMS_H

#include "Multimethod.h"
#include <array>
#include <string>
#include <vector>

enum class RecentAlgorithm {
  MadDE = 0,
  NLSHADELBC,
  SLPSOARS,
  DDEARA,
  BiPopulationCDE,
  RLPHH,
  DTGPAM,
  StandardDE,
  StandardPSO,
  StandardGA
};

const char* RecentAlgorithmName(RecentAlgorithm algorithm);
bool ParseRecentAlgorithm(const std::string& name, RecentAlgorithm& algorithm);

// Matched-budget implementations of the published search mechanisms.  The
// original equations and parameters are retained where the common CED budget
// permits; the representation, population size, stopping rule, surrogate and
// migration layer are intentionally supplied by the CED experiment driver.
class RecentSchedulingSearch {
public:
  RecentSchedulingSearch(MultiMet& solver, RecentAlgorithm algorithm);
  void Step(int generation, int max_generation, int p_start, int p_end);
  double DTGPAlpha() const { return gp_alpha_; }
  int DTGPActiveNodes() const;

private:
  struct ShadeMemory {
    std::array<double, 10> f{};
    std::array<double, 10> cr{};
    int position = 0;
  };

  static constexpr int GPTerminalCount = 11;
  static constexpr int GPMaxDepth = 8;
  static constexpr int GPNodeCount = (1 << (GPMaxDepth + 1)) - 1;
  struct GPTree {
    // 0--5: protected binary functions; 6--11: terminals; 255: inactive.
    std::array<unsigned char, GPNodeCount> node{};
  };
  struct GPIndividual {
    GPTree vmsr;
    GPTree tsr;
    double fitness = 1e300;
  };

  MultiMet& solver_;
  RecentAlgorithm algorithm_;
  ShadeMemory shade_;
  std::array<double, 3> madde_probability_{{1.0 / 3.0, 1.0 / 3.0,
                                            1.0 / 3.0}};
  std::vector<double> region_radius_;
  std::array<std::array<double, 3>, 13> rl_q_{};
  int rl_state_ = 0;
  std::vector<GPIndividual> gp_population_;
  std::array<std::array<double, GPTerminalCount>, 2> gp_terminal_weight_{};
  double gp_alpha_ = 5.0;
  double gp_smoothed_success_ = 0.2;

  void MadDEStep(int generation, int max_generation, int p_start, int p_end);
  void NLSHADELBStep(int generation, int max_generation, int p_start,
                     int p_end);
  void SLPSOARSStep(int generation, int max_generation, int p_start,
                    int p_end);
  void DDEARAStep(int generation, int max_generation, int p_start, int p_end);
  void BiPopulationCDEStep(int generation, int max_generation, int p_start,
                           int p_end);
  void RLPHHStep(int generation, int max_generation, int p_start, int p_end);
  void DTGPAMStep(int generation, int max_generation, int p_start, int p_end);
  void StandardDEStep(int generation, int max_generation, int p_start,
                      int p_end);
  void StandardPSOStep(int generation, int max_generation, int p_start,
                       int p_end);
  void StandardGAStep(int generation, int max_generation, int p_start,
                      int p_end);

  void EvaluateWithProxyPolicy(int index, int generation, int max_generation,
                               double old_fit);
  void Bound(double* candidate, const double* parent) const;
  int DistinctIndex(int avoid0, int avoid1 = -1, int avoid2 = -1) const;
  std::vector<int> FitnessOrder() const;
  double Normal(double mean, double standard_deviation) const;
  double Cauchy(double location, double scale) const;
  void GenerateGPTree(GPTree& tree, int index, int depth, int target_depth,
                      bool full, int tree_type, double sampling_ratio);
  double EvaluateGPTree(const GPTree& tree, int index,
                        const std::array<double, GPTerminalCount>& terminals)
      const;
  void SwapGPSubtree(GPTree& first, GPTree& second, int root);
  void MutateGPSubtree(GPTree& tree, int tree_type, double sampling_ratio);
  int GPTournament() const;
  void UpdateGPTerminalWeights(double sampling_ratio);
};

#endif
