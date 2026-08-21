// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef CED_SCHEDULE_MULTIMETHOD_H
#include "Multimethod.h"
#endif
#include <cmath>
#include <cstring>

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_SUBSPACE_DIM
#define ADE_SUBSPACE_DIM 48
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_PATH_UPDATE_DIM
#define ADE_PATH_UPDATE_DIM 64
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_LARGE_CROSSOVER_DIM
#define ADE_LARGE_CROSSOVER_DIM 3072
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_HUGE_CROSSOVER_DIM
#define ADE_HUGE_CROSSOVER_DIM 236
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_RANDOM_IMMIGRANTS
#define ADE_RANDOM_IMMIGRANTS 3
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_BIPOP_BLOCK_CYCLING
#define ADE_BIPOP_BLOCK_CYCLING 0
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_PROXY_ALIGNED_DIMENSIONS
#define ADE_PROXY_ALIGNED_DIMENSIONS 0
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_FACTORY_STRATIFIED_BLOCK
#define ADE_FACTORY_STRATIFIED_BLOCK 0
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_SPARSE_SUCCESS_ARCHIVE
#define ADE_SPARSE_SUCCESS_ARCHIVE 0
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_SPARSE_ARCHIVE_SIZE
#define ADE_SPARSE_ARCHIVE_SIZE 8
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ADE_SPARSE_COMMIT
#define ADE_SPARSE_COMMIT 0
#endif

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
double clamp_value(double value, double low, double high) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value < low)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return low;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value > high)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return high;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return value;
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
double normal_step(double mean, double sigma) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (sigma <= 0.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return mean;
  const double u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
  const double u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return mean + sigma * sqrt(-2.0 * log(u1)) * cos(2.0 * kPi * u2);
}

// 段落说明：复制候选或档案状态，保持父代、子代和验证值一一对应。
void copy_candidate(double* target, const double* source, int nvar) {
  std::memcpy(target, source, sizeof(double) * nvar);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
bool normalize_population_range(int popsize, int& p_start, int& p_end) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (p_start < 0)
    p_start = 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (p_end > popsize)
    p_end = popsize;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return p_start < p_end;
}

// 段落说明：实现 `ade_dim_index`：完成该函数负责的数据准备、算法步骤和状态返回。
int ade_dim_index(int nvar, int particle, int step, int salt) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return (step * 9973 + particle * 101 + salt * 4099) % nvar;
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace

// 段落说明：实现 `MultiMet::seri_diff_left_index`：完成该函数负责的数据准备、算法步骤和状态返回。
int MultiMet::seri_diff_left_index(int term) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return AdeDiffStart + term * 3;
}

// 段落说明：实现 `MultiMet::seri_diff_right_index`：完成该函数负责的数据准备、算法步骤和状态返回。
int MultiMet::seri_diff_right_index(int term) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return AdeDiffStart + term * 3 + 1;
}

// 段落说明：实现 `MultiMet::seri_diff_weight_index`：完成该函数负责的数据准备、算法步骤和状态返回。
int MultiMet::seri_diff_weight_index(int term) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return AdeDiffStart + term * 3 + 2;
}

// 段落说明：实现 `AdeTrialStats::AdeTrialStats`：完成该函数负责的数据准备、算法步骤和状态返回。
MultiMet::AdeTrialStats::AdeTrialStats() {
  reset();
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::AdeTrialStats::reset() {
  success_count = 0;
  total_improvement = 0.0;
  memory_f1_num = 0.0;
  memory_f1_den = 0.0;
  memory_f2_num = 0.0;
  memory_f2_den = 0.0;
  memory_cr_sum = 0.0;
  fill(trial_base_count, trial_base_count + AdeBaseModeCount, 0);
  fill(trial_mutation_count, trial_mutation_count + AdeMutationTypeCount, 0);
  fill(trial_diff_count, trial_diff_count + AdeMaxDiffTerms, 0);
  fill(trial_diff_mode_count, trial_diff_mode_count + AdeDiffModeCount, 0);
  fill(&trial_base_diff_count[0][0],
       &trial_base_diff_count[0][0] + AdeBaseModeCount * AdeDiffModeCount, 0);
  fill(&trial_path_count[0][0][0],
       &trial_path_count[0][0][0] +
           AdeBaseModeCount * AdeDiffModeCount * AdeMutationTypeCount,
       0);
  fill(policy_base_hits, policy_base_hits + AdeBaseModeCount, 0);
  fill(policy_mutation_hits, policy_mutation_hits + AdeMutationTypeCount, 0);
  fill(policy_diff_hits, policy_diff_hits + AdeMaxDiffTerms, 0);
  fill(policy_diff_mode_hits, policy_diff_mode_hits + AdeDiffModeCount, 0);
  fill(&policy_base_diff_hits[0][0],
       &policy_base_diff_hits[0][0] + AdeBaseModeCount * AdeDiffModeCount, 0);
  fill(&policy_path_hits[0][0][0],
       &policy_path_hits[0][0][0] +
           AdeBaseModeCount * AdeDiffModeCount * AdeMutationTypeCount,
       0);
  fill(policy_base_reward_sqrt, policy_base_reward_sqrt + AdeBaseModeCount,
       0.0);
  fill(policy_mutation_reward_sqrt,
       policy_mutation_reward_sqrt + AdeMutationTypeCount, 0.0);
  fill(policy_diff_reward_sqrt, policy_diff_reward_sqrt + AdeMaxDiffTerms, 0.0);
  fill(policy_diff_mode_reward_sqrt,
       policy_diff_mode_reward_sqrt + AdeDiffModeCount, 0.0);
  fill(&policy_base_diff_reward_sqrt[0][0],
       &policy_base_diff_reward_sqrt[0][0] +
           AdeBaseModeCount * AdeDiffModeCount,
       0.0);
  fill(&policy_path_reward_sqrt[0][0][0],
       &policy_path_reward_sqrt[0][0][0] +
           AdeBaseModeCount * AdeDiffModeCount * AdeMutationTypeCount,
       0.0);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
MultiMet::AdePolicyTables MultiMet::default_seri_policy() {
  AdePolicyTables tables = {};
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int state = 0; state < HyperStateCount; state++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeBaseModeCount; i++)
      tables.base[state][i] = 1.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeMutationTypeCount; i++)
      tables.mutation[state][i] = 1.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeMaxDiffTerms; i++)
      tables.diff[state][i] = 1.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeDiffModeCount; i++)
      tables.diff_mode[state][i] = 1.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int base = 0; base < AdeBaseModeCount; base++) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int mode = 0; mode < AdeDiffModeCount; mode++) {
        tables.base_diff[state][base][mode] = 1.0;
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int mutation = 0; mutation < AdeMutationTypeCount; mutation++)
          tables.path[state][base][mode][mutation] = 1.0;
      }
    }
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return tables;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::share_seri_policy(AdePolicyTables& tables, int source_state,
                                 double rate) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (source_state < 0 || source_state >= HyperStateCount)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int state = 0; state < HyperStateCount; state++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (state == source_state)
      continue;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeBaseModeCount; i++)
      tables.base[state][i] +=
          rate * (tables.base[source_state][i] - tables.base[state][i]);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeMutationTypeCount; i++)
      tables.mutation[state][i] +=
          rate * (tables.mutation[source_state][i] - tables.mutation[state][i]);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeMaxDiffTerms; i++)
      tables.diff[state][i] +=
          rate * (tables.diff[source_state][i] - tables.diff[state][i]);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < AdeDiffModeCount; i++)
      tables.diff_mode[state][i] += rate * (tables.diff_mode[source_state][i] -
                                            tables.diff_mode[state][i]);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int base = 0; base < AdeBaseModeCount; base++) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int mode = 0; mode < AdeDiffModeCount; mode++) {
        tables.base_diff[state][base][mode] +=
            rate * (tables.base_diff[source_state][base][mode] -
                    tables.base_diff[state][base][mode]);
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int mutation = 0; mutation < AdeMutationTypeCount; mutation++) {
          tables.path[state][base][mode][mutation] +=
              rate * (tables.path[source_state][base][mode][mutation] -
                      tables.path[state][base][mode][mutation]);
        }
      }
    }
  }
}

// 段落说明：实现 `MultiMet::normalize_seri`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::normalize_seri(double* seri, double progress) {
  const double fmax = 0.55 - 0.25 * progress;
  const int active_limit = max(1, ade_active_size);
  seri[AdeBaseMode] =
      clamp_value((int)seri[AdeBaseMode], 0, AdeBaseModeCount - 1);
  seri[AdeBaseIndex] =
      clamp_value((int)seri[AdeBaseIndex], 0, active_limit - 1);
  seri[AdeDiffCount] = clamp_value((int)seri[AdeDiffCount], 1, AdeMaxDiffTerms);
  seri[AdeDiffModeIndex] =
      clamp_value((int)seri[AdeDiffModeIndex], 0, AdeDiffModeCount - 1);
  seri[AdeCr] = clamp_value(seri[AdeCr], 0.05, 1.0);
  seri[AdeMutationType] =
      clamp_value((int)seri[AdeMutationType], 0, AdeMutationTypeCount - 1);
  seri[AdeMutationRate] = clamp_value(seri[AdeMutationRate], 0.0, 0.16);
  seri[AdeMutationScale] =
      clamp_value(seri[AdeMutationScale], 1e-6,
                  max(0.0015, 0.05 * (1.0 - progress) + 0.0015));
  seri[AdeMutationMix] = clamp_value(seri[AdeMutationMix], 0.0, 0.38);
  seri[AdePathBlend] = clamp_value(seri[AdePathBlend], 0.0, 0.65);
  seri[AdePathSource] = clamp_value((int)seri[AdePathSource], 0, 3);
  seri[AdePathScale] = clamp_value(seri[AdePathScale], 1e-6,
                                   max(0.04, 0.45 * (1.0 - progress) + 0.08));
  seri[AdePathRecentMix] = clamp_value(seri[AdePathRecentMix], 0.0, 0.45);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int term = 0; term < AdeMaxDiffTerms; term++) {
    const int left = seri_diff_left_index(term);
    const int right = seri_diff_right_index(term);
    const int weight = seri_diff_weight_index(term);
    seri[left] = clamp_value((int)seri[left], 0, Popsize + 2);
    seri[right] = clamp_value((int)seri[right], 0, Popsize + 2);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if ((int)seri[left] == (int)seri[right])
      seri[right] = ((int)seri[right] + 1) % active_limit;
    seri[weight] = clamp_value(seri[weight], 0.03, max(0.08, fmax));
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
int MultiMet::sample_left_source(int active_limit) {
  const double r = randval(0, 1);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (r < 0.45)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return Popsize;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (r < 0.65)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return Popsize + 2;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (r < 0.82)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return rand() % active_limit;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (r < 0.92)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return Popsize + 1;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return rand() % active_limit;
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
int MultiMet::sample_right_source(int active_limit) {
  const double r = randval(0, 1);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (r < 0.62)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return rand() % active_limit;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (r < 0.78)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return Popsize + 2;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (r < 0.90)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return Popsize + 1;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return Popsize;
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
bool high_error_search(double best_fit, double progress, double threshold) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return best_fit > threshold && progress > 0.25;
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
double high_error_epsilon(double progress, double base_rate, double floor_rate,
                          bool high_error) {
  const double epsilon = base_rate * (1.0 - progress);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return high_error ? max(epsilon, floor_rate) : epsilon;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void assign_path_seri(MultiMet& solver, double* seri, double progress,
                      int hyper_state) {
  const int base = clamp_value((int)seri[MultiMet::AdeBaseMode], 0,
                               MultiMet::AdeBaseModeCount - 1);
  const int mode = clamp_value((int)seri[MultiMet::AdeDiffModeIndex], 0,
                               MultiMet::AdeDiffModeCount - 1);
  const int mutation = clamp_value((int)seri[MultiMet::AdeMutationType], 0,
                                   MultiMet::AdeMutationTypeCount - 1);
  const double explore = 1.0 - progress;
  const double path_policy =
      solver.ade_policy.path[hyper_state][base][mode][mutation];
  const double path_activation =
      clamp_value((progress - 0.03) / 0.12, 0.0, 1.0);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  seri[MultiMet::AdePathBlend] =
      path_activation *
      (0.035 + 0.260 * explore * clamp_value(path_policy, 0.25, 2.5));
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (solver.gbest_fit > 100.0 && solver.randval(0.0, 1.0) < 0.72)
    seri[MultiMet::AdePathSource] = solver.randval(0.0, 1.0) < 0.55 ? 1 : 3;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    seri[MultiMet::AdePathSource] =
        (path_policy > 1.15 && solver.randval(0.0, 1.0) < 0.65) ? 0
                                                                : rand() % 4;
  seri[MultiMet::AdePathScale] =
      0.04 + solver.randval(0.0, 0.34 * explore + 0.08);
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (solver.gbest_fit < 0.0)
    seri[MultiMet::AdePathRecentMix] = 0.0;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (solver.gbest_fit < 1.0)
    seri[MultiMet::AdePathRecentMix] = 0.05;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (solver.gbest_fit > 100.0)
    seri[MultiMet::AdePathRecentMix] = 0.34;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    seri[MultiMet::AdePathRecentMix] = 0.12;
}
} // namespace

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::sample_diff_terms(double* seri, const double* mf1,
                                 const double* mf2, double progress,
                                 const double* diff_policy) {
  int diff_count = 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (randval(0.0, 1.0) < 0.82) {
    const bool high_error = high_error_search(gbest_fit, progress, 1e4);
    diff_count = choose_policy_action(diff_policy, AdeMaxDiffTerms,
                                      0.10 * (1.0 - progress), !high_error) +
                 1;
  } else {
    const double r = randval(0, 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (r > 0.90 && r <= 0.985)
      diff_count = 2;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r > 0.985 && r <= 0.997)
      diff_count = 3;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r > 0.997)
      diff_count = 4;
  }
  seri[AdeDiffCount] = diff_count;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  const int active_limit = max(1, ade_active_size);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int term = 0; term < AdeMaxDiffTerms; term++) {
    seri[seri_diff_left_index(term)] = sample_left_source(active_limit);
    seri[seri_diff_right_index(term)] = sample_right_source(active_limit);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if ((int)seri[seri_diff_left_index(term)] ==
        (int)seri[seri_diff_right_index(term)])
      seri[seri_diff_right_index(term)] = rand() % active_limit;

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    const double base_f =
        (term == 0) ? mf1[rand() % AdeMemorySize] : mf2[rand() % AdeMemorySize];
    seri[seri_diff_weight_index(term)] =
        base_f + normal_step(0.0, 0.16 * (1.0 - progress) + 0.025);
  }
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
double pair_policy_mix(const double* base_diff_policy, int count, int base_mode,
                       double progress) {
  double min_weight = base_diff_policy[0];
  double max_weight = base_diff_policy[0];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 1; i < count; i++) {
    min_weight = min(min_weight, base_diff_policy[i]);
    max_weight = max(max_weight, base_diff_policy[i]);
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  const double confidence =
      clamp_value((max_weight - min_weight - 0.08) / 0.45, 0.0, 1.0);
  double mix_scale = 1.0;
  const double late_scale = clamp_value((progress - 0.35) / 0.45, 0.0, 1.0);
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (base_mode == MultiMet::AdeBaseGlobalBest ||
      base_mode == MultiMet::AdeBaseCurrentToBest ||
      base_mode == MultiMet::AdeBaseCurrentToPBest)
    mix_scale = 1.0 - 0.45 * late_scale;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (base_mode == MultiMet::AdeBasePersonalBest)
    mix_scale = 1.0 - 0.25 * late_scale;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return mix_scale * (0.06 + 0.04 * confidence);
}
} // namespace

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::choose_diff_mode(double* seri, double progress, int base_mode,
                                const double* diff_mode_policy,
                                const double* base_diff_policy) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (randval(0.0, 1.0) < 0.84) {
    double combined_policy[AdeDiffModeCount];
    const double mix = pair_policy_mix(base_diff_policy, AdeDiffModeCount,
                                       base_mode, progress);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int mode = 0; mode < AdeDiffModeCount; mode++)
      combined_policy[mode] =
          diff_mode_policy[mode] * (1.0 - mix + mix * base_diff_policy[mode]);
    seri[AdeDiffModeIndex] = choose_policy_action(
        combined_policy, AdeDiffModeCount, 0.12 * (1.0 - progress), true);
  } else {
    const double r = randval(0, 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (r < 0.20)
      seri[AdeDiffModeIndex] = AdeDiffAdaptivePairs;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.30)
      seri[AdeDiffModeIndex] = AdeDiffRand1;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.43)
      seri[AdeDiffModeIndex] = AdeDiffBest1;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.61)
      seri[AdeDiffModeIndex] = AdeDiffCurrentToBest1;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.80)
      seri[AdeDiffModeIndex] = AdeDiffCurrentToPBest1;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.87)
      seri[AdeDiffModeIndex] = AdeDiffBeeNeighbor;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.93)
      seri[AdeDiffModeIndex] = AdeDiffRand2;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.98)
      seri[AdeDiffModeIndex] = AdeDiffElite;
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      seri[AdeDiffModeIndex] = AdeDiffOpposition;
  }
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::choose_base_strategy(double* seri, double progress,
                                    const double* base_policy) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (randval(0.0, 1.0) < 0.82) {
    const bool high_error = high_error_search(gbest_fit, progress, 5e3);
    const double epsilon = high_error_epsilon(progress, 0.12, 0.02, high_error);
    seri[AdeBaseMode] = choose_policy_action(
        base_policy, AdeBaseModeCount, epsilon, progress > 0.55 && !high_error);
  } else {
    const double r = randval(0, 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (r < 0.24 + 0.18 * progress)
      seri[AdeBaseMode] = AdeBaseCurrentToBest;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.38 + 0.18 * progress)
      seri[AdeBaseMode] = AdeBaseGlobalBest;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.50 + 0.10 * progress)
      seri[AdeBaseMode] = AdeBasePersonalBest;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.62)
      seri[AdeBaseMode] = AdeBaseCurrentToPBest;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.72)
      seri[AdeBaseMode] = AdeBaseEliteMean;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.84)
      seri[AdeBaseMode] = AdeBaseIndexedBest;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.94)
      seri[AdeBaseMode] = AdeBaseRandomBest;
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      seri[AdeBaseMode] = AdeBaseCurrent;
  }
  seri[AdeBaseIndex] = rand() % max(1, ade_active_size);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::choose_mutation_strategy(double* seri, double progress,
                                        const double* mutation_policy) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (randval(0.0, 1.0) < 0.82) {
    const bool high_error = high_error_search(gbest_fit, progress, 5e3);
    const double epsilon = high_error_epsilon(progress, 0.12, 0.02, high_error);
    seri[AdeMutationType] = choose_policy_action(
        mutation_policy, AdeMutationTypeCount, epsilon, !high_error);
  } else {
    const double r = randval(0, 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (r < 0.35 + 0.35 * progress)
      seri[AdeMutationType] = 0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.55)
      seri[AdeMutationType] = 1;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.76)
      seri[AdeMutationType] = 3;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.86)
      seri[AdeMutationType] = 2;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.90)
      seri[AdeMutationType] = 6;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (r < 0.96)
      seri[AdeMutationType] = 4;
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      seri[AdeMutationType] = 5;
  }

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  const double explore = 1.0 - progress;
  seri[AdeMutationRate] = 0.003 + randval(0.0, 0.04 + 0.08 * explore);
  seri[AdeMutationScale] = 0.001 + randval(0.0, 0.035 * explore + 0.012);
  seri[AdeMutationMix] = randval(0.02, 0.25 + 0.12 * explore);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::sample_seri(double* seri, const double* mf1, const double* mf2,
                           const double* mcr, double progress,
                           const double* base_policy,
                           const double* mutation_policy,
                           const double* diff_policy,
                           const double* diff_mode_policy,
                           const double base_diff_policy[][AdeDiffModeCount]) {
  const int m = rand() % AdeMemorySize;
  choose_base_strategy(seri, progress, base_policy);
  sample_diff_terms(seri, mf1, mf2, progress, diff_policy);
  const int base = clamp_value((int)seri[AdeBaseMode], 0, AdeBaseModeCount - 1);
  choose_diff_mode(seri, progress, base, diff_mode_policy,
                   base_diff_policy[base]);
  choose_mutation_strategy(seri, progress, mutation_policy);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  const double explore = 1.0 - progress;
  const int policy_state =
      (int)clamp_value(search_state(progress, 0), 0, HyperStateCount - 1);
  seri[seri_diff_weight_index(0)] =
      mf1[m] + normal_step(0.0, 0.18 * explore + 0.03);
  seri[seri_diff_weight_index(1)] =
      mf2[m] + normal_step(0.0, 0.18 * explore + 0.03);
  seri[AdeCr] = mcr[m] + normal_step(0.0, 0.22 * explore + 0.03);
  assign_path_seri(*this, seri, progress, policy_state);
  normalize_seri(seri, progress);
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
void limit_diff_count_by_state(double* seri, double progress, int hyper_state,
                               double best_fit) {
  int max_diff_count = 2;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (hyper_state == MultiMet::HyperStagnation)
    max_diff_count = progress < 0.80 ? 4 : 3;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (hyper_state == MultiMet::HyperEarly && progress < 0.20)
    max_diff_count = 3;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (best_fit > 1e3 && progress > 0.25)
    max_diff_count = 3;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (best_fit < 0.0)
    max_diff_count = 3;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if ((int)seri[MultiMet::AdeDiffCount] > max_diff_count)
    seri[MultiMet::AdeDiffCount] = max_diff_count;
}
} // namespace

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::sample_seri(double* seri, const double* mf1, const double* mf2,
                           const double* mcr, double progress,
                           int hyper_state) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (hyper_state < 0)
    hyper_state = 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  else if (hyper_state >= HyperStateCount)
    hyper_state = HyperStateCount - 1;
  sample_seri(seri, mf1, mf2, mcr, progress, ade_policy.base[hyper_state],
              ade_policy.mutation[hyper_state], ade_policy.diff[hyper_state],
              ade_policy.diff_mode[hyper_state],
              ade_policy.base_diff[hyper_state]);
  assign_path_seri(*this, seri, progress, hyper_state);
  limit_diff_count_by_state(seri, progress, hyper_state, gbest_fit);
  normalize_seri(seri, progress);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::reset_seri_pool(double Seri[][AdeSeriSize], const double* mf1,
                               const double* mf2, const double* mcr,
                               double progress, int hyper_state) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (hyper_state < 0)
    hyper_state = 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  else if (hyper_state >= HyperStateCount)
    hyper_state = HyperStateCount - 1;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    sample_seri(Seri[i], mf1, mf2, mcr, progress, hyper_state);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::update_ade_memory(double* mf1, double* mf2, double* mcr,
                                 int& memory_pos, const AdeTrialStats& stats) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (stats.total_improvement <= 0.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (stats.memory_f1_den > 0.0)
    mf1[memory_pos] = stats.memory_f1_num / stats.memory_f1_den;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (stats.memory_f2_den > 0.0)
    mf2[memory_pos] = stats.memory_f2_num / stats.memory_f2_den;
  mcr[memory_pos] = stats.memory_cr_sum / stats.total_improvement;
  memory_pos = (memory_pos + 1) % AdeMemorySize;
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
void normalize_policy(double* policy, int count, double floor_value,
                      double ceiling_value) {
  double mean = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int action = 0; action < count; action++)
    mean += policy[action];
  mean /= max(1, count);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (mean <= 1e-12)
    mean = 1.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int action = 0; action < count; action++)
    policy[action] =
        clamp_value(policy[action] / mean, floor_value, ceiling_value);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void update_action_policy(double* policy, int count, const int* trial_count,
                          const int* hits, const double* reward_sqrt_sum,
                          double reward_scale, double alpha,
                          double reward_weight, double floor_value,
                          double ceiling_value) {
  static vector<double> quality_by_action;
  quality_by_action.assign(count, 0.0);
  double total_trials = 0.0;
  double baseline_quality = 0.0;
  int total_hits = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int action = 0; action < count; action++) {
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (trial_count[action] <= 0)
      continue;

    // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
    const double trials = trial_count[action] + 1e-12;
    const double success_rate = hits[action] / trials;
    const double mean_reward = reward_sqrt_sum[action] * reward_scale / trials;
    const double quality = 0.30 * success_rate + reward_weight * mean_reward;
    quality_by_action[action] = quality;
    total_trials += trial_count[action];
    total_hits += hits[action];
    baseline_quality += trial_count[action] * quality;
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (total_trials > 0.0)
    baseline_quality /= total_trials;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  if (total_trials > 0.0 && total_hits == 0) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < count; action++) {
      const double tried_share =
          trial_count[action] > 0 ? trial_count[action] / total_trials : 0.0;
      policy[action] *= exp(-0.020 * tried_share);
    }
  } else {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < count; action++) {
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (trial_count[action] <= 0)
        continue;

      // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
      const double trials = trial_count[action] + 1e-12;
      const double failure_rate = (trial_count[action] - hits[action]) / trials;
      const double advantage = quality_by_action[action] - baseline_quality;
      const double confidence = sqrt(trials / (trials + 2.0));
      const double penalty =
          advantage < 0.0 ? 0.025 * confidence * failure_rate : 0.0;
      policy[action] *= exp(alpha * advantage - penalty);
    }
  }

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  normalize_policy(policy, count, floor_value, ceiling_value);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void accumulate_policy_success(int action, int count, double reward_sqrt,
                               int* hits, double* reward_sqrt_sum) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (action < 0 || action >= count)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  hits[action]++;
  reward_sqrt_sum[action] += reward_sqrt;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void update_pair_policy(
    double policy[][MultiMet::AdeDiffModeCount],
    const int trial_count[][MultiMet::AdeDiffModeCount],
    const int hits[][MultiMet::AdeDiffModeCount],
    const double reward_sqrt_sum[][MultiMet::AdeDiffModeCount],
    double reward_scale) {
  const int action_count =
      MultiMet::AdeBaseModeCount * MultiMet::AdeDiffModeCount;
  update_action_policy(&policy[0][0], action_count, &trial_count[0][0],
                       &hits[0][0], &reward_sqrt_sum[0][0], reward_scale, 0.08,
                       0.35, 0.65, 1.45);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  for (int base = 0; base < MultiMet::AdeBaseModeCount; base++)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int mode = 0; mode < MultiMet::AdeDiffModeCount; mode++)
      policy[base][mode] = clamp_value(policy[base][mode], 0.65, 1.45);
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
void update_path_policy(
    double policy[][MultiMet::AdeDiffModeCount][MultiMet::AdeMutationTypeCount],
    const int trial_count[][MultiMet::AdeDiffModeCount]
                         [MultiMet::AdeMutationTypeCount],
    const int hits[][MultiMet::AdeDiffModeCount]
                  [MultiMet::AdeMutationTypeCount],
    const double reward_sqrt_sum[][MultiMet::AdeDiffModeCount]
                                [MultiMet::AdeMutationTypeCount],
    double reward_scale) {
  const int action_count = MultiMet::AdeBaseModeCount *
                           MultiMet::AdeDiffModeCount *
                           MultiMet::AdeMutationTypeCount;
  update_action_policy(&policy[0][0][0], action_count, &trial_count[0][0][0],
                       &hits[0][0][0], &reward_sqrt_sum[0][0][0], reward_scale,
                       0.10, 0.45, 0.55, 1.65);
}
} // namespace

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::update_seri_policy(int hyper_state, const AdeTrialStats& stats) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (hyper_state < 0)
    hyper_state = 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  else if (hyper_state >= HyperStateCount)
    hyper_state = HyperStateCount - 1;
  const double reward_scale = 1.0 / sqrt(stats.total_improvement + 1e-12);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  update_action_policy(ade_policy.base[hyper_state], AdeBaseModeCount,
                       stats.trial_base_count, stats.policy_base_hits,
                       stats.policy_base_reward_sqrt, reward_scale, 0.18, 0.95,
                       0.04, 3.5);
  update_action_policy(ade_policy.mutation[hyper_state], AdeMutationTypeCount,
                       stats.trial_mutation_count, stats.policy_mutation_hits,
                       stats.policy_mutation_reward_sqrt, reward_scale, 0.16,
                       0.75, 0.04, 3.5);
  update_action_policy(ade_policy.diff[hyper_state], AdeMaxDiffTerms,
                       stats.trial_diff_count, stats.policy_diff_hits,
                       stats.policy_diff_reward_sqrt, reward_scale, 0.14, 0.65,
                       0.04, 3.5);
  update_action_policy(ade_policy.diff_mode[hyper_state], AdeDiffModeCount,
                       stats.trial_diff_mode_count, stats.policy_diff_mode_hits,
                       stats.policy_diff_mode_reward_sqrt, reward_scale, 0.16,
                       0.80, 0.04, 3.5);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (Nvar > 500000)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  update_pair_policy(ade_policy.base_diff[hyper_state],
                     stats.trial_base_diff_count, stats.policy_base_diff_hits,
                     stats.policy_base_diff_reward_sqrt, reward_scale);
  update_path_policy(ade_policy.path[hyper_state], stats.trial_path_count,
                     stats.policy_path_hits, stats.policy_path_reward_sqrt,
                     reward_scale);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::collect_ade_trial_stats(double Seri[][AdeSeriSize],
                                       unsigned char* Sflag,
                                       AdeTrialStats& stats, int p_start,
                                       int p_end) {
  stats.reset();
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (!normalize_population_range(Popsize, p_start, p_end))
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const bool collect_high_order_stats = Nvar <= 500000;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    const int trial_base = (int)Seri[i][AdeBaseMode];
    const int trial_mutation = (int)Seri[i][AdeMutationType];
    const int trial_diff = (int)Seri[i][AdeDiffCount];
    const int trial_diff_mode = (int)Seri[i][AdeDiffModeIndex];
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (trial_base >= 0 && trial_base < AdeBaseModeCount)
      stats.trial_base_count[trial_base]++;
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (trial_mutation >= 0 && trial_mutation < AdeMutationTypeCount)
      stats.trial_mutation_count[trial_mutation]++;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (trial_diff >= 1 && trial_diff <= AdeMaxDiffTerms)
      stats.trial_diff_count[trial_diff - 1]++;
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (trial_diff_mode >= 0 && trial_diff_mode < AdeDiffModeCount)
      stats.trial_diff_mode_count[trial_diff_mode]++;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (collect_high_order_stats && trial_base >= 0 &&
        trial_base < AdeBaseModeCount && trial_diff_mode >= 0 &&
        trial_diff_mode < AdeDiffModeCount)
      stats.trial_base_diff_count[trial_base][trial_diff_mode]++;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (collect_high_order_stats && trial_base >= 0 &&
        trial_base < AdeBaseModeCount && trial_diff_mode >= 0 &&
        trial_diff_mode < AdeDiffModeCount && trial_mutation >= 0 &&
        trial_mutation < AdeMutationTypeCount)
      stats.trial_path_count[trial_base][trial_diff_mode][trial_mutation]++;

    // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
    if (newpop_fit[i] < pop_fit[i]) {
      Sflag[i] = false;
      const double success_f1 = Seri[i][seri_diff_weight_index(0)];
      const double success_f2 = Seri[i][seri_diff_weight_index(1)];
      const double success_cr = Seri[i][AdeCr];
      const double improvement = pop_fit[i] - newpop_fit[i];
      const double weight = improvement;
      const double reward_sqrt = sqrt(improvement);
      stats.success_count++;
      stats.total_improvement += weight;
      accumulate_policy_success(trial_base, AdeBaseModeCount, reward_sqrt,
                                stats.policy_base_hits,
                                stats.policy_base_reward_sqrt);
      accumulate_policy_success(trial_mutation, AdeMutationTypeCount,
                                reward_sqrt, stats.policy_mutation_hits,
                                stats.policy_mutation_reward_sqrt);
      accumulate_policy_success(trial_diff - 1, AdeMaxDiffTerms, reward_sqrt,
                                stats.policy_diff_hits,
                                stats.policy_diff_reward_sqrt);
      accumulate_policy_success(trial_diff_mode, AdeDiffModeCount, reward_sqrt,
                                stats.policy_diff_mode_hits,
                                stats.policy_diff_mode_reward_sqrt);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (collect_high_order_stats && trial_base >= 0 &&
          trial_base < AdeBaseModeCount && trial_diff_mode >= 0 &&
          trial_diff_mode < AdeDiffModeCount) {
        stats.policy_base_diff_hits[trial_base][trial_diff_mode]++;
        stats.policy_base_diff_reward_sqrt[trial_base][trial_diff_mode] +=
            reward_sqrt;
        // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
        if (trial_mutation >= 0 && trial_mutation < AdeMutationTypeCount) {
          stats.policy_path_hits[trial_base][trial_diff_mode][trial_mutation]++;
          stats.policy_path_reward_sqrt[trial_base][trial_diff_mode]
                                       [trial_mutation] += reward_sqrt;
        }
      }
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (ade_path_center != nullptr && ade_recent_path != nullptr) {
        const double path_rate =
            clamp_value(0.018 + 0.055 * Seri[i][AdePathBlend], 0.018, 0.075);
        const double recent_rate = clamp_value(2.2 * path_rate, 0.04, 0.16);
        const int path_update_dim = min(Nvar, ADE_PATH_UPDATE_DIM);
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int step = 0; step < path_update_dim; step++) {
          const int dim = path_update_dim == Nvar
                              ? step
                              : (i * 101 + step * 9973) % Nvar;
          const double successful_step = newpop[i][dim] - pop[i][dim];
          ade_path_center[dim] = (1.0 - path_rate) * ade_path_center[dim] +
                                 path_rate * successful_step;
          ade_recent_path[dim] =
              (1.0 - recent_rate) * ade_recent_path[dim] +
              recent_rate * successful_step;
        }
        ade_path_ready = true;
      }
      stats.memory_f1_num += weight * success_f1 * success_f1;
      stats.memory_f1_den += weight * success_f1;
      stats.memory_f2_num += weight * success_f2 * success_f2;
      stats.memory_f2_den += weight * success_f2;
      stats.memory_cr_sum += weight * success_cr;
    } else
      Sflag[i] = true;
  }
}

// 段落说明：实现 `MultiMet::ade_active_population_size`：完成该函数负责的数据准备、算法步骤和状态返回。
int MultiMet::ade_active_population_size() const {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return ade_active_size;
}

// 段落说明：实现 `MultiMet::compact_active_population`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::compact_active_population() {
  static vector<double> pop_store;
  static vector<double> ibest_store;
  vector<int> order(Popsize);
  vector<double> pop_fit_store(Popsize);
  vector<double> ibest_fit_store(Popsize);

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  pop_store.resize(ade_active_size * Nvar);
  ibest_store.resize(ade_active_size * Nvar);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Popsize; i++)
    order[i] = i;
  sort(order.begin(), order.end(),
       [&](int a, int b) { return pop_fit[a] < pop_fit[b]; });

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < ade_active_size; i++) {
    const int source = order[i];
    copy_candidate(&pop_store[i * Nvar], pop[source], Nvar);
    copy_candidate(&ibest_store[i * Nvar], ibest[source], Nvar);
    pop_fit_store[i] = pop_fit[source];
    ibest_fit_store[i] = ibest_fit[source];
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = ade_active_size; i < Popsize; i++) {
    const int source = i % ade_active_size;
    copy_candidate(pop[i], &pop_store[source * Nvar], Nvar);
    copy_candidate(ibest[i], &ibest_store[source * Nvar], Nvar);
    pop_fit[i] = pop_fit_store[source];
    ibest_fit[i] = ibest_fit_store[source];
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < ade_active_size; i++) {
    copy_candidate(pop[i], &pop_store[i * Nvar], Nvar);
    copy_candidate(ibest[i], &ibest_store[i * Nvar], Nvar);
    pop_fit[i] = pop_fit_store[i];
    ibest_fit[i] = ibest_fit_store[i];
  }
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
void MultiMet::update_shade_population_size(double progress,
                                            int stagnant_generations,
                                            const AdeTrialStats& stats) {
  const int min_size = max(8, Popsize / 2);
  int target = (int)round(Popsize - (Popsize - min_size) * progress);

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (stagnant_generations > 60 && stats.success_count == 0)
    target = max(target, ade_active_size - 1);

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (target < min_size)
    target = min_size;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  else if (target > Popsize)
    target = Popsize;
  target = min(target, ade_active_size);
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (target < ade_active_size) {
    ade_active_size = target;
    compact_active_population();
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::adapt_successful_seri(double* seri, const double* mf1,
                                     const double* mf2, const double* mcr,
                                     double progress) {
  const double explore = 1.0 - progress;
  const int m = rand() % AdeMemorySize;
  const double memory_mix = 0.12 + 0.18 * explore;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  if (randval(0.0, 1.0) < 0.10 * explore)
    seri[AdeBaseIndex] = rand() % max(1, ade_active_size);

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  if (randval(0.0, 1.0) < 0.12 * explore)
    seri[AdeDiffCount] = 1 + rand() % 2;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (randval(0.0, 1.0) < 0.08 * explore)
    seri[AdeDiffModeIndex] = rand() % AdeDiffModeCount;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  seri[seri_diff_weight_index(0)] =
      (1.0 - memory_mix) * seri[seri_diff_weight_index(0)] +
      memory_mix * (mf1[m] + normal_step(0.0, 0.04 * explore + 0.008));
  seri[seri_diff_weight_index(1)] =
      (1.0 - memory_mix) * seri[seri_diff_weight_index(1)] +
      memory_mix * (mf2[m] + normal_step(0.0, 0.04 * explore + 0.008));
  seri[AdeCr] =
      (1.0 - memory_mix) * seri[AdeCr] +
      memory_mix * (mcr[m] + normal_step(0.0, 0.06 * explore + 0.008));

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  if (randval(0.0, 1.0) < 0.08 * explore)
    seri[AdeMutationType] = rand() % AdeMutationTypeCount;
  seri[AdeMutationRate] += normal_step(0.0, 0.009 * explore + 0.0015);
  seri[AdeMutationScale] += normal_step(0.0, 0.003 * explore + 0.0006);
  seri[AdeMutationMix] += normal_step(0.0, 0.025 * explore + 0.003);

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  normalize_seri(seri, progress);
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
int clamp_index(double value, int low, int high) {
  const int index = (int)value;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (index < low)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return low;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (index > high)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return high;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return index;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct AdeCandidateContext {
  int base_mode;
  int diff_count;
  int diff_mode;
  int active_limit;
  int mutation_type;
  double diff_scale;
  double diff_weight[MultiMet::AdeMaxDiffTerms];
  double bee_phi;
  double mutation_rate;
  double mutation_scale;
  double mutation_mix;
  double crossover_rate;
  double path_blend;
  double path_scale;
  double path_recent_mix;
  int path_source;
  double lower_bound;
  double upper_bound;
  double bound_sum;
  double bound_range;
  const double* parent;
  const double* personal_best;
  const double* global_best;
  const double* pbest;
  const double* random_best;
  const double* indexed_best;
  const double* r1_best;
  const double* r2_best;
  const double* r3_best;
  const double* r4_best;
  const double* bee_neighbor_best;
  const double* elite_a_best;
  const double* elite_b_best;
  const double* base_vector;
  const double* path_center;
  const double* recent_path;
  const double* diff_left_value[MultiMet::AdeMaxDiffTerms];
  const double* diff_right_value[MultiMet::AdeMaxDiffTerms];
};

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
int random_other_index(int active_limit, int avoid) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (active_limit <= 1)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  int index = rand() % active_limit;
  // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
  while (index == avoid)
    index = rand() % active_limit;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return index;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
const double* ade_source_pointer(MultiMet& solver, int candidate, int source,
                                 int active_limit) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (source >= 0 && source < solver.ade_active_size)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return solver.ibest[source];
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (source >= solver.ade_active_size && source < solver.Popsize)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return solver.ibest[source % active_limit];
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (source == solver.Popsize)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return solver.gbest;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (source == solver.Popsize + 1)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return solver.pop[candidate];
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (source == solver.Popsize + 2)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return solver.ibest[candidate];
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return solver.pop[candidate];
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void collect_elite_indices(MultiMet& solver, int active_limit,
                           vector<int>& elite_indices) {
  const int elite_count = min(active_limit, max(1, active_limit / 5));
  elite_indices.clear();
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if ((int)elite_indices.capacity() < elite_count)
    elite_indices.reserve(elite_count);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int candidate = 0; candidate < active_limit; candidate++) {
    int pos = 0;
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    while (pos < (int)elite_indices.size() &&
           solver.ibest_fit[elite_indices[pos]] <= solver.ibest_fit[candidate])
      pos++;
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (pos < elite_count) {
      elite_indices.insert(elite_indices.begin() + pos, candidate);
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if ((int)elite_indices.size() > elite_count)
        elite_indices.pop_back();
    }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
AdeCandidateContext make_ade_context(MultiMet& solver, double* seri, int index,
                                     const vector<int>& elite_indices) {
  AdeCandidateContext ctx;
  ctx.active_limit = max(1, solver.ade_active_size);
  ctx.base_mode = clamp_index(seri[MultiMet::AdeBaseMode], 0,
                              MultiMet::AdeBaseModeCount - 1);
  const int base_index =
      clamp_index(seri[MultiMet::AdeBaseIndex], 0, ctx.active_limit - 1);
  ctx.diff_count =
      clamp_index(seri[MultiMet::AdeDiffCount], 1, MultiMet::AdeMaxDiffTerms);
  ctx.diff_mode = clamp_index(seri[MultiMet::AdeDiffModeIndex], 0,
                              MultiMet::AdeDiffModeCount - 1);
  ctx.diff_scale = 1.0 / sqrt((double)ctx.diff_count);
  ctx.lower_bound = solver.Lbound;
  ctx.upper_bound = solver.Ubound;
  ctx.bound_sum = solver.Lbound + solver.Ubound;
  ctx.bound_range = fabs((double)(solver.Ubound - solver.Lbound));
  const int pbest_index = elite_indices[rand() % elite_indices.size()];
  const int r1 = rand() % ctx.active_limit;
  const int r2 = random_other_index(ctx.active_limit, r1);
  const int r3 = random_other_index(ctx.active_limit, r2);
  const int r4 = random_other_index(ctx.active_limit, r3);
  const int random_best_index = rand() % ctx.active_limit;
  const int bee_neighbor =
      random_other_index(ctx.active_limit, index % ctx.active_limit);
  ctx.bee_phi = solver.randval(-1.0, 1.0);
  const int elite_a = elite_indices[rand() % elite_indices.size()];
  const int elite_b = elite_indices[rand() % elite_indices.size()];
  ctx.parent = solver.pop[index];
  ctx.personal_best = solver.ibest[index];
  ctx.global_best = solver.gbest;
  ctx.pbest = solver.ibest[pbest_index];
  ctx.random_best = solver.ibest[random_best_index];
  ctx.indexed_best = solver.ibest[base_index];
  ctx.r1_best = solver.ibest[r1];
  ctx.r2_best = solver.ibest[r2];
  ctx.r3_best = solver.ibest[r3];
  ctx.r4_best = solver.ibest[r4];
  ctx.bee_neighbor_best = solver.ibest[bee_neighbor];
  ctx.elite_a_best = solver.ibest[elite_a];
  ctx.elite_b_best = solver.ibest[elite_b];
  ctx.base_vector = 0;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (ctx.base_mode == MultiMet::AdeBasePersonalBest)
    ctx.base_vector = ctx.personal_best;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (ctx.base_mode == MultiMet::AdeBaseGlobalBest)
    ctx.base_vector = ctx.global_best;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (ctx.base_mode == MultiMet::AdeBaseRandomBest)
    ctx.base_vector = ctx.random_best;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  else if (ctx.base_mode == MultiMet::AdeBaseIndexedBest)
    ctx.base_vector = ctx.indexed_best;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (ctx.base_mode == MultiMet::AdeBaseEliteMean)
    ctx.base_vector = ctx.pbest;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (ctx.base_mode == MultiMet::AdeBaseCurrent)
    ctx.base_vector = ctx.parent;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int term = 0; term < MultiMet::AdeMaxDiffTerms; term++) {
    int diff_left = clamp_index(seri[MultiMet::seri_diff_left_index(term)], 0,
                                solver.Popsize + 2);
    int diff_right = clamp_index(seri[MultiMet::seri_diff_right_index(term)], 0,
                                 solver.Popsize + 2);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (diff_left == diff_right)
      diff_right = (diff_right + 1) % ctx.active_limit;
    ctx.diff_weight[term] = seri[MultiMet::seri_diff_weight_index(term)];
    ctx.diff_left_value[term] =
        ade_source_pointer(solver, index, diff_left, ctx.active_limit);
    ctx.diff_right_value[term] =
        ade_source_pointer(solver, index, diff_right, ctx.active_limit);
  }
  ctx.mutation_type = (int)seri[MultiMet::AdeMutationType];
  ctx.mutation_rate = seri[MultiMet::AdeMutationRate];
  ctx.mutation_scale = seri[MultiMet::AdeMutationScale] * ctx.bound_range;
  ctx.mutation_mix = seri[MultiMet::AdeMutationMix];
  ctx.crossover_rate = seri[MultiMet::AdeCr];
  ctx.path_blend = seri[MultiMet::AdePathBlend];
  ctx.path_source = clamp_index(seri[MultiMet::AdePathSource], 0, 3);
  ctx.path_scale = seri[MultiMet::AdePathScale];
  ctx.path_center = solver.ade_path_ready ? solver.ade_path_center : 0;
  ctx.recent_path = solver.ade_path_ready ? solver.ade_recent_path : 0;
  ctx.path_recent_mix = seri[MultiMet::AdePathRecentMix];
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return ctx;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
double build_ade_base_value(const AdeCandidateContext& ctx, int dim) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ctx.base_vector)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.base_vector[dim];

  // 段落说明：实现 `switch`：完成该函数负责的数据准备、算法步骤和状态返回。
  switch (ctx.base_mode) {
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeBaseCurrentToBest:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.parent[dim] +
           ctx.diff_weight[0] * (ctx.global_best[dim] - ctx.parent[dim]);
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeBaseCurrentToPBest:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.parent[dim] +
           ctx.diff_weight[0] * (ctx.pbest[dim] - ctx.parent[dim]);
  default:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.parent[dim];
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
double ade_diff_delta(const AdeCandidateContext& ctx, int dim,
                      double current_value) {
  // 控制说明：根据算法/动作枚举分派到对应实现，避免不同方法共享错误路径。
  switch (ctx.diff_mode) {
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffAdaptivePairs: {
    double delta = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int term = 0; term < ctx.diff_count; term++) {
      delta +=
          ctx.diff_scale * ctx.diff_weight[term] *
          (ctx.diff_left_value[term][dim] - ctx.diff_right_value[term][dim]);
    }
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return delta;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffRand1:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.diff_weight[0] * (ctx.r1_best[dim] - ctx.r2_best[dim]);
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffBest1:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.5 * ctx.diff_weight[0] * (ctx.global_best[dim] - current_value) +
           ctx.diff_weight[1] * (ctx.r1_best[dim] - ctx.r2_best[dim]);
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffCurrentToBest1:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.diff_weight[0] * (ctx.global_best[dim] - ctx.parent[dim]) +
           ctx.diff_weight[1] * (ctx.r1_best[dim] - ctx.r2_best[dim]);
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffCurrentToPBest1:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.diff_weight[0] * (ctx.pbest[dim] - ctx.parent[dim]) +
           ctx.diff_weight[1] * (ctx.r1_best[dim] - ctx.r2_best[dim]);
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffRand2:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.diff_scale *
           (ctx.diff_weight[0] * (ctx.r1_best[dim] - ctx.r2_best[dim]) +
            ctx.diff_weight[1] * (ctx.r3_best[dim] - ctx.r4_best[dim]));
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffElite:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.diff_weight[0] * (ctx.elite_a_best[dim] - ctx.elite_b_best[dim]);
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffOpposition: {
    const double opposite = ctx.bound_sum - ctx.r1_best[dim];
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.diff_weight[0] * (opposite - ctx.r2_best[dim]);
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case MultiMet::AdeDiffBeeNeighbor:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ctx.diff_weight[0] * ctx.bee_phi *
           (ctx.bee_neighbor_best[dim] - ctx.parent[dim]);
  default:
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.0;
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
double repair_ade_bound(const AdeCandidateContext& ctx, int dim, double value) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value > ctx.upper_bound)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.5 * (ctx.upper_bound + ctx.parent[dim]);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value < ctx.lower_bound)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.5 * (ctx.lower_bound + ctx.parent[dim]);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return value;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
double mutate_ade_value(MultiMet& solver, const AdeCandidateContext& ctx,
                        int dim, double value) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ctx.mutation_type == 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return repair_ade_bound(ctx, dim, value);

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (ctx.mutation_type == 1) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (solver.randval(0.0, 1.0) < ctx.mutation_rate)
      value += normal_step(0.0, ctx.mutation_scale);
  } else if (ctx.mutation_type == 2) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (solver.randval(0.0, 1.0) < ctx.mutation_rate) {
      double cauchy = tan(kPi * (solver.randval(0.0, 1.0) - 0.5));
      cauchy = clamp_value(cauchy, -4.0, 4.0);
      value += ctx.mutation_scale * cauchy;
    }
  } else if (ctx.mutation_type == 3) {
    value += ctx.mutation_mix * (ctx.global_best[dim] - value);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (solver.randval(0.0, 1.0) < ctx.mutation_rate)
      value += normal_step(0.0, ctx.mutation_scale * 0.25);
  } else if (ctx.mutation_type == 4) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (solver.randval(0.0, 1.0) < ctx.mutation_rate)
      value = solver.randval(ctx.lower_bound, ctx.upper_bound);
  } else if (ctx.mutation_type == 5) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (solver.randval(0.0, 1.0) < ctx.mutation_rate) {
      const double opposite = ctx.bound_sum - value;
      value = 0.5 * value + 0.5 * opposite +
              normal_step(0.0, ctx.mutation_scale * 0.15);
    }
  } else if (ctx.mutation_type == 6) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (solver.randval(0.0, 1.0) < ctx.mutation_rate) {
      const double local_phi = solver.randval(-1.0, 1.0);
      value +=
          ctx.mutation_mix * local_phi * (ctx.bee_neighbor_best[dim] - value);
    }
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  return repair_ade_bound(ctx, dim, value);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
double apply_policy_path_value(MultiMet& solver, const AdeCandidateContext& ctx,
                               int dim, double value) {
  (void)solver;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ctx.path_blend <= 0.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return repair_ade_bound(ctx, dim, value);

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  double direction = 0.0;
  double noise = 0.0;
  double path_scale = ctx.path_scale;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ctx.path_source == 0 && ctx.path_center) {
    const double recent_mix = ctx.recent_path ? ctx.path_recent_mix : 0.0;
    direction = (1.0 - recent_mix) * ctx.path_center[dim];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (ctx.recent_path)
      direction += recent_mix * ctx.recent_path[dim];
  } else if (ctx.path_source == 1)
    direction = ctx.global_best[dim] - ctx.parent[dim];
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (ctx.path_source == 2)
    direction = ctx.personal_best[dim] - ctx.parent[dim];
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (ctx.path_source == 3)
    direction = ctx.pbest[dim] - ctx.parent[dim];
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    direction = ctx.global_best[dim] - ctx.parent[dim];

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  value += ctx.path_blend * path_scale * direction + ctx.path_blend * noise;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return repair_ade_bound(ctx, dim, value);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
double build_ade_trial_value(MultiMet& solver, const AdeCandidateContext& ctx,
                             int dim) {
  double value = build_ade_base_value(ctx, dim);
  value += ade_diff_delta(ctx, dim, value);
  value = mutate_ade_value(solver, ctx, dim, value);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return apply_policy_path_value(solver, ctx, dim, value);
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void configure_random_immigrant_path(MultiMet& solver, double* seri) {
  const int active_limit = max(1, solver.ade_active_size);
  seri[MultiMet::AdeBaseMode] = MultiMet::AdeBaseCurrent;
  seri[MultiMet::AdeBaseIndex] = rand() % active_limit;
  seri[MultiMet::AdeDiffCount] = 1;
  seri[MultiMet::AdeCr] = 1.0;
  seri[MultiMet::AdeMutationType] = 4;
  seri[MultiMet::AdeMutationRate] = 1.0;
  seri[MultiMet::AdeMutationScale] = 0.01;
  seri[MultiMet::AdeMutationMix] = 0.0;
  seri[MultiMet::AdeDiffModeIndex] = MultiMet::AdeDiffAdaptivePairs;
  seri[MultiMet::AdePathBlend] = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int term = 0; term < MultiMet::AdeMaxDiffTerms; term++) {
    seri[MultiMet::seri_diff_left_index(term)] = rand() % active_limit;
    seri[MultiMet::seri_diff_right_index(term)] = rand() % active_limit;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if ((int)seri[MultiMet::seri_diff_left_index(term)] ==
        (int)seri[MultiMet::seri_diff_right_index(term)])
      seri[MultiMet::seri_diff_right_index(term)] =
          ((int)seri[MultiMet::seri_diff_right_index(term)] + 1) % active_limit;
    seri[MultiMet::seri_diff_weight_index(term)] = 0.20;
  }
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::build_ade_candidate(double* Seri, int i,
                                   const vector<int>& elite_indices) {
  const AdeCandidateContext ctx =
      make_ade_context(*this, Seri, i, elite_indices);
  const int forced_dim = rand() % Nvar;
  double* trial_vector = newpop[i];
  const double* parent = pop[i];

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  if (Nvar <= 100000) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; j++)
      trial_vector[j] =
          (j == forced_dim || randval(0.0, 1.0) <= ctx.crossover_rate)
              ? build_ade_trial_value(*this, ctx, j)
              : parent[j];
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  }

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  const int crossover_dim = min(Nvar, Nvar <= 500000 ? ADE_LARGE_CROSSOVER_DIM
                                                     : ADE_HUGE_CROSSOVER_DIM);
  int block_begin = 0;
  int block_size = Nvar;
// 控制说明：选择当前编译配置对应的实现路径。
#if ADE_BIPOP_BLOCK_CYCLING
  if (CE_Tnum > 0 && M_Jnum > 0 && M_OPTnum > 0) {
    const int operations = M_Jnum * M_OPTnum;
    const int block = (ade_generation + i) & 3;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (block == 0) {
      block_begin = 0;
      block_size = CE_Tnum;
    } else if (block == 1) {
      block_begin = CE_Tnum;
      block_size = CE_Tnum;
    } else if (block == 2) {
      block_begin = 2 * CE_Tnum;
      block_size = operations;
    } else {
      block_begin = 2 * CE_Tnum + operations;
      block_size = operations;
    }
  }
#endif
  block_size = max(1, min(block_size, Nvar - block_begin));
  const int local_forced_dim = block_begin + rand() % block_size;
  trial_vector[local_forced_dim] =
      build_ade_trial_value(*this, ctx, local_forced_dim);
  const int offset = rand() % block_size;
  vector<int> changed_dims;
  vector<unsigned char> changed_flags;
  changed_dims.reserve(crossover_dim + 1);
  changed_flags.reserve(crossover_dim + 1);
  changed_dims.push_back(local_forced_dim);
  changed_flags.push_back(1);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int step = 0; step < crossover_dim; step++) {
    int j = block_begin +
            (ade_dim_index(block_size, i, step, 83) + offset) % block_size;
// 控制说明：选择当前编译配置对应的实现路径。
#if ADE_PROXY_ALIGNED_DIMENSIONS
    if (step < min(crossover_dim, 96) && CE_Tnum > 0 && M_Jnum > 0) {
      const int group = step & 3;
      const int sample = step >> 2;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (group < 2) {
        const int stride = max(1, CE_Tnum / 256);
        const int sample_count = (CE_Tnum + stride - 1) / stride;
        const int local =
            ((ade_generation * 29 + sample * 17 + i * 7) % sample_count) *
            stride;
        j = (group == 0 ? 0 : CE_Tnum) + min(local, CE_Tnum - 1);
      } else {
        const int operations = M_Jnum * M_OPTnum;
        const int stride = max(1, operations / 512);
        const int sample_count = (operations + stride - 1) / stride;
        const int local =
            ((ade_generation * 31 + sample * 19 + i * 11) % sample_count) *
            stride;
        const int base = 2 * CE_Tnum + (group == 3 ? operations : 0);
        j = base + min(local, operations - 1);
      }
    }
#endif
    const bool changed = randval(0.0, 1.0) <= ctx.crossover_rate;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (changed)
      trial_vector[j] = build_ade_trial_value(*this, ctx, j);
    changed_dims.push_back(j);
    changed_flags.push_back(changed ? 1 : 0);
  }

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if ADE_FACTORY_STRATIFIED_BLOCK
  if (CE_Tnum >= 1000 && M_Jnum == CE_Tnum && M_OPTnum > 0) {
    const int tasks_per_factory = 1000;
    const int factories = CE_Tnum / tasks_per_factory;
    const int local_task =
        (ade_generation * 37 + i * 101 + 83) % tasks_per_factory;
    const int operation_count = M_Jnum * M_OPTnum;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int factory = 0; factory < factories; ++factory) {
      const int task = factory * tasks_per_factory + local_task;
      const int task_dims[2] = {task, CE_Tnum + task};
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int d = 0; d < 2; ++d) {
        const int dim = task_dims[d];
        const bool changed = randval(0.0, 1.0) <= ctx.crossover_rate;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (changed)
          trial_vector[dim] = build_ade_trial_value(*this, ctx, dim);
        changed_dims.push_back(dim);
        changed_flags.push_back(changed ? 1 : 0);
      }
      const int operation_base = task * M_OPTnum;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int operation = 0; operation < M_OPTnum; ++operation) {
        const int local_operation = operation_base + operation;
        const int operation_dims[2] = {
            2 * CE_Tnum + local_operation,
            2 * CE_Tnum + operation_count + local_operation};
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int d = 0; d < 2; ++d) {
          const int dim = operation_dims[d];
          const bool changed = randval(0.0, 1.0) <= ctx.crossover_rate;
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (changed)
            trial_vector[dim] = build_ade_trial_value(*this, ctx, dim);
          changed_dims.push_back(dim);
          changed_flags.push_back(changed ? 1 : 0);
        }
      }
    }
  }
#endif

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  if ((int)ade_pending_dimensions.size() != Popsize) {
    ade_pending_dimensions.resize(Popsize);
    ade_pending_previous_values.resize(Popsize);
  }
  vector<int>& pending_dims = ade_pending_dimensions[i];
  vector<double>& pending_values = ade_pending_previous_values[i];
  pending_dims.clear();
  pending_values.clear();
  pending_dims.reserve(changed_dims.size());
  pending_values.reserve(changed_dims.size());
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int k = 0; k < (int)changed_dims.size(); ++k) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!changed_flags[k])
      continue;
    pending_dims.push_back(changed_dims[k]);
    pending_values.push_back(parent[changed_dims[k]]);
  }

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if ADE_SPARSE_SUCCESS_ARCHIVE
  if ((int)ade_sparse_archive.size() == Popsize &&
      !ade_sparse_archive[i].empty()) {
    const int source_count = (int)ade_sparse_archive[i].size() + 1;
    const int source_a = rand() % source_count;
    int source_b = rand() % (source_count - 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (source_b >= source_a)
      ++source_b;
    const double archive_f =
        clamp_value(std::fabs(ctx.diff_weight[0]), 0.10, 0.90);

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    for (int k = 0; k < (int)pending_dims.size(); ++k) {
      const int dim = pending_dims[k];
      double value_a = parent[dim];
      double value_b = parent[dim];
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int age = 0; age < max(source_a, source_b); ++age) {
        const AdeSparseArchiveEntry& entry =
            ade_sparse_archive[i][ade_sparse_archive[i].size() - 1 - age];
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int p = 0; p < (int)entry.dimensions.size(); ++p) {
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (entry.dimensions[p] != dim)
            continue;
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (age < source_a)
            value_a = entry.previous_values[p];
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (age < source_b)
            value_b = entry.previous_values[p];
          break;
        }
      }
      trial_vector[dim] = repair_ade_bound(
          ctx, dim, trial_vector[dim] + archive_f * (value_a - value_b));
    }
  }
#endif
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::ADE(double Seri[][AdeSeriSize], int p_start, int p_end) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (!normalize_population_range(ade_active_size, p_start, p_end))
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  static vector<int> elite_indices;
  collect_elite_indices(*this, max(1, ade_active_size), elite_indices);
  const int candidate_count = p_end - p_start;
  const int immigrant_count =
      min(ADE_RANDOM_IMMIGRANTS, max(0, candidate_count - 1));
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (i >= p_end - immigrant_count)
      configure_random_immigrant_path(*this, Seri[i]);
    build_ade_candidate(Seri[i], i, elite_indices);
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::EnsureADERuntime(double progress, int hyper_state) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if ((int)ade_seri_pool.size() != Popsize * AdeSeriSize)
    ade_runtime_ready = false;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (ade_runtime_ready)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  ade_seri_pool.assign(Popsize * AdeSeriSize, 0.0);
  ade_success_flags.assign(Popsize, 1);
  ade_sparse_archive.assign(Popsize, vector<AdeSparseArchiveEntry>());
  ade_pending_dimensions.assign(Popsize, vector<int>());
  ade_pending_previous_values.assign(Popsize, vector<double>());
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < AdeMemorySize; i++) {
    ade_mf1[i] = 0.45;
    ade_mf2[i] = 0.28;
    ade_mcr[i] = 0.88;
  }
  ade_memory_pos = 0;
  ade_stagnant_generations = 0;
  ade_generation = 0;
  ade_last_best = gbest_fit;
  ade_active_size = Popsize;
  ade_policy = default_seri_policy();
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ade_path_center != nullptr && ade_recent_path != nullptr)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Nvar; i++) {
      ade_path_center[i] = 0.0;
      ade_recent_path[i] = 0.0;
    }
  ade_path_ready = false;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  double (*seri)[AdeSeriSize] =
      reinterpret_cast<double (*)[AdeSeriSize]>(ade_seri_pool.data());
  reset_seri_pool(seri, ade_mf1, ade_mf2, ade_mcr, progress, hyper_state);
  ade_runtime_ready = true;
}

// 段落说明：实现 `MultiMet::ADE`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::ADE(int gen, int max_gen, int p_start, int p_end) {
  double progress = max_gen > 0 ? (double)gen / (double)max_gen : 1.0;
  progress = clamp_value(progress, 0.0, 1.0);
  int hyper_state = search_state(progress, ade_stagnant_generations);
  EnsureADERuntime(progress, hyper_state);
  ade_generation = gen;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  double (*seri)[AdeSeriSize] =
      reinterpret_cast<double (*)[AdeSeriSize]>(ade_seri_pool.data());
  ADE(seri, p_start, p_end);
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
void MultiMet::FinalizeADEGeneration(int gen, int max_gen, int p_start,
                                     int p_end) {
  double progress = max_gen > 0 ? (double)(gen + 1) / (double)max_gen : 1.0;
  progress = clamp_value(progress, 0.0, 1.0);
  int hyper_state = search_state(progress, ade_stagnant_generations);
  EnsureADERuntime(progress, hyper_state);

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  double (*seri)[AdeSeriSize] =
      reinterpret_cast<double (*)[AdeSeriSize]>(ade_seri_pool.data());
  AdeTrialStats stats;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if ((int)ade_success_flags.size() != Popsize)
    ade_success_flags.assign(Popsize, 1);
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    fill(ade_success_flags.begin(), ade_success_flags.end(), 1);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  collect_ade_trial_stats(seri, ade_success_flags.data(), stats, p_start,
                          p_end);
// 控制说明：选择当前编译配置对应的实现路径。
#if ADE_SPARSE_SUCCESS_ARCHIVE
  for (int i = p_start; i < p_end; ++i) {
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (ade_success_flags[i] || ade_pending_dimensions[i].empty())
      continue;
    AdeSparseArchiveEntry entry;
    entry.dimensions.swap(ade_pending_dimensions[i]);
    entry.previous_values.swap(ade_pending_previous_values[i]);
    vector<AdeSparseArchiveEntry>& archive = ade_sparse_archive[i];
    archive.push_back(std::move(entry));
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if ((int)archive.size() > ADE_SPARSE_ARCHIVE_SIZE)
      archive.erase(archive.begin());
  }
#endif
  update_ade_memory(ade_mf1, ade_mf2, ade_mcr, ade_memory_pos, stats);
  update_seri_policy(hyper_state, stats);
  const bool has_success = stats.success_count > 0;
  share_seri_policy(ade_policy, hyper_state, has_success ? 0.011 : 0.014);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const double success_ratio =
      (double)stats.success_count / max(1, p_end - p_start);
  const double keep_success_probability =
      clamp_value(0.52 + 0.25 * progress + 0.25 * success_ratio, 0.45, 0.92);
  const int resample_state = has_success ? hyper_state : HyperStagnation;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!ade_success_flags[i] && randval(0.0, 1.0) < keep_success_probability)
      adapt_successful_seri(seri[i], ade_mf1, ade_mf2, ade_mcr, progress);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      sample_seri(seri[i], ade_mf1, ade_mf2, ade_mcr, progress, resample_state);
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const bool defer_success_commit =
      p_start == 0 && p_end == Popsize && Popsize <= 8;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (defer_success_commit) {
    vector<unsigned char> gbest_updated_sparsely(Popsize, 0);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = p_start; i < p_end; i++) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (ade_success_flags[i]) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (Nvar > 100000 &&
            i < (int)ade_pending_dimensions.size() &&
            !ade_pending_dimensions[i].empty()) {
          // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
          for (int dim : ade_pending_dimensions[i])
            newpop[i][dim] = pop[i][dim];
        } else {
          copy_candidate(newpop[i], pop[i], Nvar);
        }
// 控制说明：选择当前编译配置对应的实现路径。
#if ADE_SPARSE_COMMIT
      } else if (Nvar > 100000 &&
                 i < (int)ade_pending_dimensions.size() &&
                 !ade_pending_dimensions[i].empty()) {
        const double old_pop_fit = pop_fit[i];
        const bool ibest_matches_parent =
            std::fabs(ibest_fit[i] - old_pop_fit) <=
            1e-12 * (1.0 + std::fabs(old_pop_fit));
        const bool gbest_matches_parent =
            std::fabs(gbest_fit - old_pop_fit) <=
            1e-12 * (1.0 + std::fabs(old_pop_fit));
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int dim : ade_pending_dimensions[i])
          pop[i][dim] = newpop[i][dim];
        pop_fit[i] = newpop_fit[i];
        // 控制说明：依据目标值决定接受、最优更新或审计路径。
        if (newpop_fit[i] < ibest_fit[i]) {
          // 控制说明：依据目标值决定接受、最优更新或审计路径。
          if (ibest_matches_parent) {
            // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
            for (int dim : ade_pending_dimensions[i])
              ibest[i][dim] = newpop[i][dim];
          } else {
            copy_candidate(ibest[i], newpop[i], Nvar);
          }
          ibest_fit[i] = newpop_fit[i];
        }
        // 控制说明：依据目标值决定接受、最优更新或审计路径。
        if (newpop_fit[i] < gbest_fit && gbest_matches_parent) {
          // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
          for (int dim : ade_pending_dimensions[i])
            gbest[dim] = newpop[i][dim];
          gbest_fit = newpop_fit[i];
          gbest_updated_sparsely[i] = 1;
        }
#endif
      }
// 控制说明：选择当前编译配置对应的实现路径。
#if ADE_SPARSE_COMMIT
      newpop_fit[i] = pop_fit[i];
// 控制说明：选择当前编译配置对应的实现路径。
#else
      newpop_fit[i] = ade_success_flags[i] ? pop_fit[i] : newpop_fit[i];
#endif
      if (i < (int)ade_pending_dimensions.size()) {
        ade_pending_dimensions[i].clear();
        ade_pending_previous_values[i].clear();
      }
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    int best_index = p_start;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = p_start + 1; i < p_end; i++)
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (newpop_fit[i] < newpop_fit[best_index])
        best_index = i;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (!gbest_updated_sparsely[best_index] &&
        newpop_fit[best_index] < gbest_fit) {
      copy_candidate(gbest, newpop[best_index], Nvar);
      gbest_fit = newpop_fit[best_index];
    }
  } else {
    pop_better_update(p_start, p_end);
    worst_and_best();
    copy_candidate(gbest, pop[cur_best], Nvar);
    gbest_fit = pop_fit[cur_best];
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  update_shade_population_size(progress, ade_stagnant_generations, stats);
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (gbest_fit < ade_last_best) {
    ade_last_best = gbest_fit;
    ade_stagnant_generations = 0;
  } else {
    ade_stagnant_generations++;
  }
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ade_stagnant_generations > 0 && ade_stagnant_generations % 80 == 0)
    reset_seri_pool(seri, ade_mf1, ade_mf2, ade_mcr, progress, HyperStagnation);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  if (!defer_success_commit) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = p_start; i < p_end; i++) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (ade_success_flags[i])
        copy_candidate(newpop[i], pop[i], Nvar);
      newpop_fit[i] = pop_fit[i];
    }
  }
}
