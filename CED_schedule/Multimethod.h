// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef CED_SCHEDULE_MULTIMETHOD_H
#define CED_SCHEDULE_MULTIMETHOD_H

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include "Population.h"
#include "Problems.h"
#include <algorithm>
#include <cassert>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>
#include <stdlib.h>
#include <string>
#include <vector>
using namespace std;

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
typedef double (*FF)(double* var, int Cnum, int Enum, int Dnum, int CE_Tnum,
                     int M_Jnum, int M_OPTnum, CETask* CETask_Property,
                     double* MTask_Time, DistanceValue** EtoD_Distance,
                     DistanceValue** DtoD_Distance,
                     vector<int>* AvailDeviceList, double* EnergyList,
                     vector<int>* CloudDevice, vector<int>* EdgeDevices,
                     vector<int>* CloudLoad, vector<int>* EdgeLoad,
                     vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
                     map<int, double>* Edge_Device_comm, double** ST,
                     double** ET, double* CE_ST, double* CE_ET);

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
enum class SearchAlg {
  Adaptive = 0,
  GA,
  NGA,
  LGA,
  VAGA,
  EAGA,
  PSO,
  CPSO,
  SPSO,
  CMPSO,
  APSO1,
  APSO2,
  APSO3,
  APSO4,
  APSO5,
  OLPSO,
  HS,
  ACO,
  COA,
  DE,
  CSA,
  ABCA,
  POA,
  ILS,
  VNS,
  GRASP,
  PBILC,
  BATA,
  FA,
  CMAES,
  ADE,
  BA
};

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
class MultiMet : public Population<double> {
public:
  MultiMet(int psize, int nn, double lb, double ub, int c_num, int e_num,
           int d_num, int ce_tnum, int m_jnum, int m_optnum, FF evaluate);
  ~MultiMet();

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
public:
  FF EvaluFunc;

  // Prob
  int Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum;
  CETask* CETask_Property;
  double* MTask_Time;
  DistanceValue** EtoD_Distance;
  DistanceValue** DtoD_Distance;
  vector<double> EdgeX, EdgeY, DeviceX, DeviceY;
  vector<int>* AvailDeviceList;
  double* EnergyList;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  vector<int>* CloudDevices;
  vector<int>* EdgeDevices;
  vector<int>* CloudLoad;
  vector<int>* EdgeLoad;
  vector<int>* DeviceLoad;
  vector<int>* CETask_coDevice;
  map<int, double>* Edge_Device_comm;
  double** ST;
  double** ET;
  double* CE_ST;
  double* CE_ET;

  // PSO
  double** ibest;
  double* ibest_fit;
  double** velocity;
  double ac1, ac2;
  double *AC1, *AC2, *AW;
  int OArow;
  int** OA;

  // ACO                      //int tao_size;
  double** ant_tao;

  // ABCA
  int* trial;
  double* pr;

  // VNS
  int* neigh;

  // CMAES parameter
  bool CMAisdone;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  int lambda;
  int mu;
  double mucov;
  double mueff;
  double* weights;
  double damps;
  double cs;
  double ccumcov;
  double ccov;
  double* xstart;
  double* stddev;
  double diagonalCov;
  // CMAES

  //! Step size.
  double sigma;
  //! Mean x vector, "parent".
  double* xmean;

  //! Sorting index of sample population.
  int* index;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  double chiN;
  //! Lower triangular matrix: i>=j for C[i][j].
  double** C;
  //! Matrix with normalize eigenvectors in columns.
  double** B;
  //! Axis lengths.
  double* rgD;
  //! Anisotropic evolution path (for covariance).
  double* pcc;
  //! Isotropic evolution path (for step length).
  double* ps;
  //! Last mean.
  double* xold;
  //! B*D*z.
  double* BDz;
  //! Temporary (random) vector used in different places.
  double* tempRandom;

  // PBILc
  double num_of_elite;
  double* PB_center;
  double* PB_sigma;

  // BatA
  double* BAT_r;
  double* BAT_A;
  double** BAT_v;

  // FA
  double* I;  // Intensity
  int* Index; // sort of fireflies according to fitness values
  double alpha;

  // BA
  int ne, nre, stlim;
  double ngh_decay, ngh_origin;
  int* ngh_decay_count;
  double* ngh;

  // meme number
  int* Ind_meme;
  double** SubDecBase;
  vector<double> ssals_step;
  vector<double> ssals_history_sum;
  vector<int> ssals_history_count;
  double ssals_minbs;
  int ssals_minbs_dim;
  double meme_gaussian_sigma;
  unsigned long meme_gaussian_trials;

  // adaptive global search
  vector<int> SearchCount;
  vector<double> SearchReward;
  vector<double> SearchRecentReward;
  vector<int> SearchStaleCount;
  int SearchTotal;

  // ADE adaptive seri state
  int ade_active_size;
  enum {
    AdeMemorySize = 8,
    AdeSeriSize = 25,
    AdeMaxDiffTerms = 4,
    AdeBaseModeCount = 8,
    AdeDiffModeCount = 9,
    AdeMutationTypeCount = 7,
    HyperStateCount = 4
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  enum AdeBaseStrategy {
    AdeBaseCurrent = 0,
    AdeBasePersonalBest = 1,
    AdeBaseGlobalBest = 2,
    AdeBaseRandomBest = 3,
    AdeBaseCurrentToBest = 4,
    AdeBaseIndexedBest = 5,
    AdeBaseCurrentToPBest = 6,
    AdeBaseEliteMean = 7
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  enum AdeDiffMode {
    AdeDiffAdaptivePairs = 0,
    AdeDiffRand1 = 1,
    AdeDiffBest1 = 2,
    AdeDiffCurrentToBest1 = 3,
    AdeDiffCurrentToPBest1 = 4,
    AdeDiffRand2 = 5,
    AdeDiffElite = 6,
    AdeDiffOpposition = 7,
    AdeDiffBeeNeighbor = 8
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  enum HyperState {
    HyperEarly = 0,
    HyperMiddle = 1,
    HyperLate = 2,
    HyperStagnation = 3
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  enum AdeSeriIndex {
    AdeBaseMode = 0,
    AdeBaseIndex = 1,
    AdeDiffCount = 2,
    AdeCr = 3,
    AdeMutationType = 4,
    AdeMutationRate = 5,
    AdeMutationScale = 6,
    AdeMutationMix = 7,
    AdeDiffModeIndex = 8,
    AdeDiffStart = 9,
    AdePathBlend = 21,
    AdePathSource = 22,
    AdePathScale = 23,
    AdePathRecentMix = 24
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  struct AdePolicyTables {
    double base[HyperStateCount][AdeBaseModeCount];
    double mutation[HyperStateCount][AdeMutationTypeCount];
    double diff[HyperStateCount][AdeMaxDiffTerms];
    double diff_mode[HyperStateCount][AdeDiffModeCount];
    double base_diff[HyperStateCount][AdeBaseModeCount][AdeDiffModeCount];
    double path[HyperStateCount][AdeBaseModeCount][AdeDiffModeCount]
               [AdeMutationTypeCount];
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  enum {
    MemePolicyActionCount = 8,
    MemeConfigurationActionCount = 4,
    MemeScopeActionCount = 8,
    MemeProxyPolicyActionCount = 2
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  struct MemePolicyTables {
    double configuration[HyperStateCount][MemeConfigurationActionCount];
    double scope[HyperStateCount][MemeScopeActionCount];
    double meme[HyperStateCount][MemePolicyActionCount];
    double proxy[HyperStateCount][MemeProxyPolicyActionCount];
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  struct AdeTrialStats {
    int success_count;
    int trial_base_count[AdeBaseModeCount];
    int trial_mutation_count[AdeMutationTypeCount];
    int trial_diff_count[AdeMaxDiffTerms];
    int trial_diff_mode_count[AdeDiffModeCount];
    int trial_base_diff_count[AdeBaseModeCount][AdeDiffModeCount];
    int trial_path_count[AdeBaseModeCount][AdeDiffModeCount]
                        [AdeMutationTypeCount];
    int policy_base_hits[AdeBaseModeCount];
    int policy_mutation_hits[AdeMutationTypeCount];
    int policy_diff_hits[AdeMaxDiffTerms];
    int policy_diff_mode_hits[AdeDiffModeCount];
    int policy_base_diff_hits[AdeBaseModeCount][AdeDiffModeCount];
    int policy_path_hits[AdeBaseModeCount][AdeDiffModeCount]
                        [AdeMutationTypeCount];
    double total_improvement;
    double memory_f1_num;
    double memory_f1_den;
    double memory_f2_num;
    double memory_f2_den;
    double memory_cr_sum;
    double policy_base_reward_sqrt[AdeBaseModeCount];
    double policy_mutation_reward_sqrt[AdeMutationTypeCount];
    double policy_diff_reward_sqrt[AdeMaxDiffTerms];
    double policy_diff_mode_reward_sqrt[AdeDiffModeCount];
    double policy_base_diff_reward_sqrt[AdeBaseModeCount][AdeDiffModeCount];
    double policy_path_reward_sqrt[AdeBaseModeCount][AdeDiffModeCount]
                                  [AdeMutationTypeCount];

    // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
    AdeTrialStats();
    void reset();
  };

  // 段落说明：定义本模块使用的类型、状态或配置数据结构。
  struct AdeSparseArchiveEntry {
    vector<int> dimensions;
    vector<double> previous_values;
  };

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  AdePolicyTables ade_policy;
  double* ade_path_center;
  double* ade_recent_path;
  bool ade_path_ready;
  vector<double> ade_seri_pool;
  vector<unsigned char> ade_success_flags;
  vector<vector<AdeSparseArchiveEntry>> ade_sparse_archive;
  vector<vector<int>> ade_pending_dimensions;
  vector<vector<double>> ade_pending_previous_values;
  double ade_mf1[AdeMemorySize];
  double ade_mf2[AdeMemorySize];
  double ade_mcr[AdeMemorySize];
  int ade_memory_pos;
  int ade_stagnant_generations;
  int ade_generation;
  double ade_last_best;
  bool ade_runtime_ready;
  MemePolicyTables meme_policy;
  int configuration_policy_trials[HyperStateCount]
                                 [MemeConfigurationActionCount];
  int scope_policy_trials[HyperStateCount][MemeScopeActionCount];
  int meme_policy_trials[HyperStateCount][MemePolicyActionCount];
  int proxy_policy_trials[HyperStateCount][MemeProxyPolicyActionCount];
  int meme_verified_successes[MemePolicyActionCount];
  int meme_stagnant_generations;
  double meme_last_best;
  int forced_meme_action;
  int current_meme_configuration;
  int current_meme_scope;

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
public:
  double randnorm(double miu, double score);
  void pop_update(int p_start, int p_end);
  void pop_better_update(int p_start, int p_end);
  void meme_state_update(int p_start, int p_end);
  void Initial();
  void Evaluation(bool evaluate_new_population, int p_start, int p_end);
  void RunGlobalSearch(int configured_alg, int generation, int max_generation,
                       int p_start, int p_end);
  void RunSelectedGlobalSearch(SearchAlg algorithm, int generation,
                               int max_generation, int p_start, int p_end,
                               bool single_individual_mode);
  void RunAdaptiveGlobalSearch(int generation, int max_generation, int p_start,
                               int p_end);
  void ResetAdaptiveGlobalSearch();
  SearchAlg ConfiguredSearchAlg(int configured_alg);
  SearchAlg SelectAdaptiveGlobalAlgorithm();
  void UpdateAdaptiveGlobalAlgorithm(SearchAlg algorithm, double reward);

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  static int seri_diff_left_index(int term);
  static int seri_diff_right_index(int term);
  static int seri_diff_weight_index(int term);
  static int search_state(double progress, int stagnant_generations);
  static AdePolicyTables default_seri_policy();
  static MemePolicyTables default_meme_policy();
  static void share_seri_policy(AdePolicyTables& policy, int source_state,
                                double rate);
  static void update_ade_memory(double* mf1, double* mf2, double* mcr,
                                int& memory_pos, const AdeTrialStats& stats);
  int choose_policy_action(const double* policy, int count, double epsilon,
                           bool squared_weight);
  void normalize_seri(double* seri, double progress);
  int sample_left_source(int active_limit);
  int sample_right_source(int active_limit);
  void sample_diff_terms(double* seri, const double* mf1, const double* mf2,
                         double progress, const double* diff_policy);
  void choose_diff_mode(double* seri, double progress, int base_mode,
                        const double* diff_mode_policy,
                        const double* base_diff_policy);
  void choose_base_strategy(double* seri, double progress,
                            const double* base_policy);
  void choose_mutation_strategy(double* seri, double progress,
                                const double* mutation_policy);
  void sample_seri(double* seri, const double* mf1, const double* mf2,
                   const double* mcr, double progress,
                   const double* base_policy, const double* mutation_policy,
                   const double* diff_policy, const double* diff_mode_policy,
                   const double base_diff_policy[][AdeDiffModeCount]);
  void sample_seri(double* seri, const double* mf1, const double* mf2,
                   const double* mcr, double progress, int hyper_state);
  void reset_seri_pool(double Seri[][AdeSeriSize], const double* mf1,
                       const double* mf2, const double* mcr, double progress,
                       int hyper_state);
  void update_seri_policy(int hyper_state, const AdeTrialStats& stats);
  void collect_ade_trial_stats(double Seri[][AdeSeriSize], unsigned char* Sflag,
                               AdeTrialStats& stats, int p_start, int p_end);
  int ade_active_population_size() const;
  void compact_active_population();
  void ResetMemePolicyState();
  int MemePolicyState(int gen, int max_gen, double& progress);
  int SelectPolicyMemeAction(int hyper_state, double progress);
  int SelectPolicyProxyAction(int hyper_state, double progress);
  void update_shade_population_size(double progress, int stagnant_generations,
                                    const AdeTrialStats& stats);
  void adapt_successful_seri(double* seri, const double* mf1, const double* mf2,
                             const double* mcr, double progress);
  void build_ade_candidate(double* Seri, int index,
                           const vector<int>& elite_indices);
  void ADE(double Seri[][AdeSeriSize], int p_start, int p_end);
  void EnsureADERuntime(double progress, int hyper_state);
  void FinalizeADEGeneration(int gen, int max_gen, int p_start, int p_end);

  // Prob
  void show_result(double* results);

  // GA
  void select(int p_start, int p_end);
  void crossover(double pc, int p_start, int p_end);
  void xover(int, int);
  void mutate(double pm, int p_start, int p_end);
  void GA(double pc, double pm, int p_start, int p_end);
  void NGA(double pc, double pm, double mdis, int p_start, int p_end);
  void LGA(double pc, double pm, double step, int nst, int p_start, int p_end);
  void VAGA(int p_start, int p_end);
  void EAGA(int p_start, int p_end);

  // PSO
  void PSO(double w, double c1, double c2, double max_ve, int p_start,
           int p_end);
  void CPSO(double w, double c1, double c2, double max_ve, int p_start,
            int p_end);
  void Subgradient(double* theta, int q, double c_step, double* subgrad);
  void SPSO(double w, double c1, double c2, double max_ve, int Gen, int MaxGen,
            int p_start, int p_end);
  void Cauchy_mutation(double* pp, int Gen, int MaxGen);
  void CMPSO(double w, double c1, double c2, double max_ve, int Gen, int MaxGen,
             int p_start, int p_end);
  void APSO_1(double c1, double c2, double max_ve, int Gen, int MaxGen,
              int p_start, int p_end);
  void APSO_2(double c1, double c2, double max_ve, int p_start, int p_end);
  void APSO_3(double max_ve, int p_start, int p_end);
  void APSO_4(double max_ve, int Gen, int MaxGen, int p_start, int p_end);
  void APSO_5(double max_ve, int Gen, int MaxGen, int p_start, int p_end);
  void CreateOA();
  void Orthogonal_P(double* P0, double w, double c, int ppn);
  void OLPSO(double w, double c, double max_ve, int p_start, int p_end);

  // HS
  void newpop_worst_best(int& w, int& b, int p_start, int p_end);
  void HS(double srate, double trate, double bw, int p_start, int p_end);

  // ACO
  void path_finding(double epsl, int p_start, int p_end);
  void phe_updating(int p_start, int p_end);
  void ACO(double epsl, int p_start, int p_end);

  // COA
  void chaos(double** incpop, int chaos_n);
  void COA(int chaos_n, int p_start, int p_end);

  // DE
  void differential_mutate(double F, int S, int p_start, int p_end);
  void differential_crossover(double cr, int p_start, int p_end);
  void DE(double F, int S, double cr, int p_start, int p_end);

  // CSA
  void levy_cuckoo(int p_start, int p_end);
  void nest_discover(double pa, int p_start, int p_end);
  void CSA(double pa, int p_start, int p_end);

  // ABCA
  void EmployedBee(int pn, double* tmp, double** pp);
  void OnlookerBee(int p_start, int p_end);
  void ScoutBee(int limit, int p_start, int p_end);
  void ABCA(int limit, int p_start, int p_end);

  // POA
  void plant_growth(int p_start, int p_end);
  void POA(int p_start, int p_end);

  // ILS
  void localsearch(double* pp, int nst);
  void ILS(int nst, int p_start, int p_end);

  // VNS
  void VNS(int nst, int p_start, int p_end);

  // GRASP
  void GRASP(double alfa, int nst, int p_start, int p_end);

  // PBILc
  void PBILC(int p_start, int p_end, double learn_rate);

  // BATA
  void BATA(int gen, int p_start, int p_end);

  // FA
  void FA(double gama, double alpha, double betamin, int gen, int p_start,
          int p_end);

  // sorting selection
  void pop_heap_adjust(int s, int len);
  void pop_heap_sort(int len);
  void newpop_heap_adjust(int s, int len);
  void newpop_heap_sort(int len);

  // niche
  void pop_niche(double mdis, int p_start, int p_end);
  // local
  void newpop_localstep(double step, int nst, int p_start, int p_end);
  // CMA-ES
  void CMAES(int gen, int p_start, int p_end);
  void CMAES_parametersetting();
  void CMAES_initial();
  void CMAES_sampleGenerate(int firstrun, int p_start, int p_end);
  void updateEigensystem();
  void eigen(double* diag, double** Q, double* rgtmp);
  void householder(double** V, double* d, double* e);
  void ql(double* d, double* e, double** V);
  double* CMAES_updateDistribution(int gen);
  void sortIndex(const double* rgFunVal, int* iindex, int n);
  void adaptC2(const int hsig, int gen);

  // BA
  void ADE(int gen, int max_gen, int p_start, int p_end);
  void BA(int p_start, int p_end);
  void NeighborFlowerPatch(int nr, int point);

  // Discrete Gradient
  void Direct_Change(int ChangeSize);

  // local search operators
  void newpop_lightweight_meme(int popi, int iterations, double scale);
  void newpop_elite_line_search(int popi, int iterations, double scale);
  void newpop_multi_scale_gaussian(int popi, int iterations, double scale);
  void newpop_random_subspace_pattern(int popi, int iterations, double scale);
  void newpop_cauchy_basin_hop(int popi, int iterations, double scale);
  void newpop_opposition_elite_blend(int popi, int iterations, double scale);
  void newpop_bit_climbing(int popi, int iterations, double scale);
  void newpop_stepsize_adaptive(int popi, int iterations, double scale);

  // memetic adaptive selection
  void ResetSSALSState();
  void meme_selection(int popi, int meme_id, double scale, int iterations);
  void RunMemeSearchStrategy(int mode, int Gen, int MaxG, double scale,
                             int p_start, int p_end);
  void meme_random_walk(double scale, int p_start, int p_end);
  void meme_simple_random(double scale, int p_start, int p_end);
  void meme_randperm(double scale, int p_start, int p_end);
  void meme_inheritance(double scale, int p_start, int p_end);
  void meme_subprob_decomposition(int Gen, int MaxG, int kk, double scale,
                                  int p_start, int p_end);
  void meme_policy_guidance(int Gen, int MaxG, double scale, int p_start,
                            int p_end);
};
#endif
