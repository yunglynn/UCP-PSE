/* UCP-PSE sparse generative coding and uniform control policy: sparse masks,
 * directional/local operators, task guidance, proxy actions, acceptance, and
 * cost-aware policy/Robbins--Monro updates. */
#include "Multimethod.h"

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <random>
#include <unordered_map>
#include <vector>

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
constexpr int kMemeIter = 1;
constexpr int kMemeOperatorCount = MultiMet::MemePolicyActionCount;
// 控制说明：选择当前编译配置对应的实现路径。
#ifndef MEME_GREEDY_CONFIGURATION_POLICY
#define MEME_GREEDY_CONFIGURATION_POLICY 1
#endif
#ifndef MEME_POLICY_SQUARED_WEIGHT
#define MEME_POLICY_SQUARED_WEIGHT 0
#endif
#ifndef TRI_POLICY_RANDOM_ABLATION
#define TRI_POLICY_RANDOM_ABLATION 0
#endif
#ifndef TRI_POLICY_FULL_COMPONENT_RANDOM_ABLATION
#define TRI_POLICY_FULL_COMPONENT_RANDOM_ABLATION 0
#endif
#ifndef PAPER_COMPONENT_POLICY
#define PAPER_COMPONENT_POLICY 1
#endif
#ifndef FORCE_GREEDY_CONFIGURATION_ACTION
#define FORCE_GREEDY_CONFIGURATION_ACTION -1
#endif
#ifndef FORCE_GREEDY_SCOPE_ACTION
#define FORCE_GREEDY_SCOPE_ACTION -1
#endif
constexpr bool kMemePolicySquaredWeight = MEME_POLICY_SQUARED_WEIGHT != 0;

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
int RandomMemeOperatorId() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return rand() % kMemeOperatorCount;
}

// 段落说明：实现 `IsRetainedMemeOperator`：完成该函数负责的数据准备、算法步骤和状态返回。
bool IsRetainedMemeOperator(int meme_id) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return meme_id >= 0 && meme_id < kMemeOperatorCount;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
double meme_clamp_value(double value, double low, double high) {
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

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
static double meme_policy_reward_from_fit(double old_fit, double new_fit,
                                          int true_eval_cost,
                                          double work_cost) {
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (!std::isfinite(old_fit) || !std::isfinite(new_fit) || new_fit >= old_fit)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(work_cost) || work_cost < 1.0)
    work_cost = 1.0;
  const double scale = fabs(old_fit) + 1e-12;
  const double relative_gain = (old_fit - new_fit) / scale;
  const double cost = sqrt(work_cost + (double)max(0, true_eval_cost));
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return sqrt(max(0.0, relative_gain)) / cost;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
static void update_policy_cell(double& policy_value, int trials,
                               double reward) {
  const double rate = 1.0 / sqrt((double)max(1, trials));
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (reward > 0.0)
    policy_value += rate * reward;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    policy_value *= 1.0 - 0.25 * rate;
  policy_value = meme_clamp_value(policy_value, 0.04, 3.5);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
static double exp3_exploration_rate(const int* trials, int action_count) {
  int observations = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int action = 0; action < action_count; action++)
    observations += trials[action];
  const double horizon = static_cast<double>(max(1, observations));
  const double numerator = log(static_cast<double>(action_count));
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return min(1.0, numerator / horizon);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
static int verified_success_count(const MultiMet& solver) {
  int count = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int action = 0; action < MultiMet::MemePolicyActionCount; ++action)
    count += solver.meme_verified_successes[action];
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return count;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
static bool has_sufficient_policy_evidence(const MultiMet& solver) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (solver.Popsize > 1)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return true;
  const double actions = static_cast<double>(MultiMet::MemePolicyActionCount);
  const int minimum_successes =
      static_cast<int>(ceil(sqrt(actions * log(actions))));
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return verified_success_count(solver) > minimum_successes;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
static int choose_uniform_action(MultiMet& solver, int action_count) {
  double weights[MultiMet::MemeScopeActionCount];
  std::fill_n(weights, action_count, 1.0);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return solver.choose_policy_action(weights, action_count, 1.0, false);
}

// 段落说明：实现 `ssals_initial_step`：完成该函数负责的数据准备、算法步骤和状态返回。
double ssals_initial_step(const MultiMet& solver) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 0.5 * (solver.Ubound - solver.Lbound);
}

// 段落说明：实现 `ssals_step_index`：完成该函数负责的数据准备、算法步骤和状态返回。
size_t ssals_step_index(const MultiMet& solver, int popi, int dim) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return static_cast<size_t>(popi) * static_cast<size_t>(solver.Nvar) +
         static_cast<size_t>(dim);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void recompute_ssals_minbs(MultiMet& solver) {
  double best_mean = 1e300;
  int best_dim = -1;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int dim = 0; dim < solver.Nvar; dim++) {
    int count = solver.ssals_history_count[dim];
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (count <= 0)
      continue;
    double mean = solver.ssals_history_sum[dim] / count;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (mean < best_mean) {
      best_mean = mean;
      best_dim = dim;
    }
  }
  solver.ssals_minbs = best_dim >= 0 ? min(0.1, best_mean) : 0.1;
  solver.ssals_minbs_dim = best_dim;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void update_ssals_history(MultiMet& solver, int dim, double effective_step) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (dim < 0 || dim >= solver.Nvar || effective_step <= 0.0 ||
      !std::isfinite(effective_step))
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  int& count = solver.ssals_history_count[dim];
  solver.ssals_history_sum[dim] += effective_step;
  count++;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  double mean = solver.ssals_history_sum[dim] / count;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (solver.ssals_minbs_dim < 0 || mean < solver.ssals_minbs ||
      dim == solver.ssals_minbs_dim)
    recompute_ssals_minbs(solver);
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
bool eval_sparse_meme_candidate(MultiMet& solver, int popi, const int* dims,
                                const double* values, int count,
                                bool dimensions_are_unique = false);
bool eval_sparse_meme_fit(MultiMet& solver, int popi, const int* dims,
                          const double* values, int count, double& fit);
} // namespace

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
MultiMet::MemePolicyTables MultiMet::default_meme_policy() {
  MemePolicyTables tables = {};
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int state = 0; state < HyperStateCount; state++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemeConfigurationActionCount; action++)
      tables.configuration[state][action] = 1.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemeScopeActionCount; action++)
      tables.scope[state][action] = 1.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemePolicyActionCount; action++)
      tables.meme[state][action] = 1.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemeProxyPolicyActionCount; action++)
      tables.proxy[state][action] = 1.0;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return tables;
}

// 段落说明：实现 `MultiMet::ResetSSALSState`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::ResetSSALSState() {
  const double init_step = ssals_initial_step(*this);
  ssals_step.assign(static_cast<size_t>(Popsize) * static_cast<size_t>(Nvar),
                    init_step);
  ssals_history_sum.assign(Nvar, 0.0);
  ssals_history_count.assign(Nvar, 0);
  ssals_minbs = 0.1;
  ssals_minbs_dim = -1;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::newpop_lightweight_meme(int popi, int L, double scale) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  const int ce_offset = CE_Tnum;
  const int seq_offset = CE_Tnum * 2;
  const int dev_offset = CE_Tnum * 2 + M_Jnum * M_OPTnum;
// 控制说明：选择当前编译配置对应的实现路径。
#ifndef GREEDY_CONFIGURATION_BLEND
#define GREEDY_CONFIGURATION_BLEND 1.0
#endif
  const double blend = MEME_GREEDY_CONFIGURATION_POLICY
                           ? GREEDY_CONFIGURATION_BLEND
                           : 0.35;
// 控制说明：选择当前编译配置对应的实现路径。
#ifndef TASK_BLOCK_BATCH_SIZE
#define TASK_BLOCK_BATCH_SIZE 1
#endif
#ifndef FACTORY_STRATIFIED_TASK_BLOCK
#if TNUM >= 1000
#define FACTORY_STRATIFIED_TASK_BLOCK 1
#else
#define FACTORY_STRATIFIED_TASK_BLOCK 0
#endif
#endif
  int task_block_count = std::max(1, TASK_BLOCK_BATCH_SIZE);
// 控制说明：选择当前编译配置对应的实现路径。
#if defined(FACTORY_STRATIFIED_TASK_BLOCK) && FACTORY_STRATIFIED_TASK_BLOCK
  const int tasks_per_factory = 1000;
  const int factory_count = std::max(1, CE_Tnum / tasks_per_factory);
  task_block_count = factory_count;
// 控制说明：选择当前编译配置对应的实现路径。
#if MEME_GREEDY_CONFIGURATION_POLICY
#if TNUM < 10000
  static const int scope_blocks[MemeScopeActionCount] =
      {512, 640, 768, 896, 1000, 1000, 1000, 1000};
// 控制说明：选择当前编译配置对应的实现路径。
#else
  static const int scope_blocks[MemeScopeActionCount] =
      {112, 144, 176, 208, 240, 272, 304, 336};
#endif
  const int blocks_per_factory =
      scope_blocks[std::max(0, std::min(MemeScopeActionCount - 1,
                                       current_meme_scope))];
  task_block_count *= blocks_per_factory;
// 控制说明：选择当前编译配置对应的实现路径。
#else
  const int blocks_per_factory = 1;
#endif
#endif

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int trial_iter = 0; trial_iter < L; trial_iter++) {
    vector<int> changed;
    vector<double> values;
    changed.reserve(task_block_count * (2 + 2 * M_OPTnum));
    values.reserve(task_block_count * (2 + 2 * M_OPTnum));
// 控制说明：选择当前编译配置对应的实现路径。
#if defined(FACTORY_STRATIFIED_TASK_BLOCK) && FACTORY_STRATIFIED_TASK_BLOCK
    const int task_stride = std::max(1, tasks_per_factory /
        std::max(1, blocks_per_factory));
    const int local_task_start = rand() % task_stride;
#endif
    auto current_value = [&](int idx) { return newpop[popi][idx]; };
    auto set_changed = [&](int idx, double value) {
      changed.push_back(idx);
      values.push_back(value);
    };

    // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
    for (int block = 0; block < task_block_count; ++block) {
      int ce_task = rand() % CE_Tnum;
// 控制说明：选择当前编译配置对应的实现路径。
#if defined(FACTORY_STRATIFIED_TASK_BLOCK) && FACTORY_STRATIFIED_TASK_BLOCK
      const int factory = block % factory_count;
      const int within_batch = block / factory_count;
      ce_task = factory * tasks_per_factory + local_task_start +
                within_batch * task_stride;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (ce_task >= CE_Tnum)
        ce_task %= CE_Tnum;
// 控制说明：选择当前编译配置对应的实现路径。
#if MEME_GREEDY_CONFIGURATION_POLICY
      const int segment_end = std::min(
          CE_Tnum, factory * tasks_per_factory +
                       (within_batch + 1) * task_stride);
      double best_importance = -1.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int candidate = ce_task; candidate < segment_end; ++candidate) {
        const double candidate_relation =
            CETask_Property[candidate].Precedence.size() +
            CETask_Property[candidate].Interact.size() +
            CETask_Property[candidate].Start_Pre.size() +
            CETask_Property[candidate].End_Pre.size();
        const double importance =
            CETask_Property[candidate].Computation / 199.0 +
            CETask_Property[candidate].Communication / 4999.0 +
            candidate_relation / 8.0;
        // 控制说明：依据目标值决定接受、最优更新或审计路径。
        if (importance > best_importance) {
          best_importance = importance;
          ce_task = candidate;
        }
      }
#endif
#endif
      double ce_value = current_value(ce_task);
// 控制说明：选择当前编译配置对应的实现路径。
#if MEME_GREEDY_CONFIGURATION_POLICY
      const int configuration_action =
          std::max(0, std::min(MemeConfigurationActionCount - 1,
                               current_meme_configuration));
      const double comp = CETask_Property[ce_task].Computation / 199.0;
      const double comm = CETask_Property[ce_task].Communication / 4999.0;
      const double relation =
          (CETask_Property[ce_task].Precedence.size() +
           CETask_Property[ce_task].Interact.size() +
           CETask_Property[ce_task].Start_Pre.size() +
           CETask_Property[ce_task].End_Pre.size()) / 8.0;
      const double task_position = static_cast<double>(ce_task) /
                                   std::max(1, CE_Tnum - 1);
      double greedy_task = 0.0;
      // 控制说明：根据算法/动作枚举分派到对应实现，避免不同方法共享错误路径。
      switch (configuration_action) {
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 1: greedy_task = 1.0; break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 2:
        greedy_task = comp + relation > comm + 0.5 ? 0.0 : 1.0;
        break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 3:
        greedy_task =
            1.0 / (1.0 + std::exp(-3.0 * (comm + relation - comp)));
        break;
      default: break;
      }
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (configuration_action == 0) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (randval(0.0, 1.0) < 0.35)
          ce_value = 1.0 - ce_value;
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else
          ce_value = (1.0 - 0.35) * ce_value + 0.35 * gbest[ce_task];
      } else {
        ce_value = (1.0 - blend) * ce_value + blend * greedy_task;
      }
// 控制说明：选择当前编译配置对应的实现路径。
#else
      if (randval(0.0, 1.0) < 0.35)
        ce_value = 1.0 - ce_value;
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        ce_value = (1.0 - blend) * ce_value + blend * gbest[ce_task];
#endif
      set_changed(ce_task, ce_value);

      // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
      int ce_server_idx = ce_offset + ce_task;
// 控制说明：选择当前编译配置对应的实现路径。
#if MEME_GREEDY_CONFIGURATION_POLICY
      const double resource_score =
          configuration_action == 3 ? comp + 0.5 * comm + relation
                                    : comm + 0.5 * comp - relation;
      static const double configuration_bias[MemeConfigurationActionCount] =
          {0.0, 0.173, 0.519, 1.038};
      double greedy_resource =
          resource_score + 0.61803398875 * task_position +
          configuration_bias[configuration_action];
      greedy_resource -= std::floor(greedy_resource);
      double ce_server_value = current_value(ce_server_idx);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (configuration_action == 0) {
        ce_server_value += randval(-scale, scale);
        ce_server_value =
            0.65 * ce_server_value + 0.35 * gbest[ce_server_idx];
      } else {
        ce_server_value = (1.0 - blend) * ce_server_value +
                          blend * greedy_resource;
      }
// 控制说明：选择当前编译配置对应的实现路径。
#else
      double ce_server_value =
          current_value(ce_server_idx) + randval(-scale, scale);
      ce_server_value =
          (1.0 - blend) * ce_server_value + blend * gbest[ce_server_idx];
#endif
      set_changed(ce_server_idx, ce_server_value);

      // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
      int job = rand() % M_Jnum;
// 控制说明：选择当前编译配置对应的实现路径。
#if defined(FACTORY_STRATIFIED_TASK_BLOCK) && FACTORY_STRATIFIED_TASK_BLOCK
      job = ce_task;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (job >= M_Jnum)
        job %= M_Jnum;
#endif
      for (int op = 0; op < M_OPTnum; op++) {
        int idx = job * M_OPTnum + op;
        int seq_idx = seq_offset + idx;
        int dev_idx = dev_offset + idx;

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if MEME_GREEDY_CONFIGURATION_POLICY
        const double duration = MTask_Time[idx] / 300.0;
        const double operation_position = static_cast<double>(idx) /
                                          std::max(1, M_Jnum * M_OPTnum - 1);
        const double stage = static_cast<double>(op) /
                             std::max(1, M_OPTnum - 1);
        double greedy_sequence = operation_position;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (configuration_action == 1)
          greedy_sequence = duration;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        else if (configuration_action == 2)
          greedy_sequence = 0.75 * stage + 0.25 * duration;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        else if (configuration_action == 3)
          greedy_sequence = duration + 0.5 * operation_position + stage;
        greedy_sequence -= std::floor(greedy_sequence);
        double seq_value = current_value(seq_idx);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (configuration_action == 0) {
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (randval(0.0, 1.0) < 0.6)
            seq_value = 0.65 * seq_value + 0.35 * gbest[seq_idx];
          // 控制说明：条件不成立时执行互斥的备用处理路径。
          else
            seq_value += randval(-scale, scale);
        } else {
          seq_value = (1.0 - blend) * seq_value + blend * greedy_sequence;
        }
// 控制说明：选择当前编译配置对应的实现路径。
#else
        double seq_value = current_value(seq_idx);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (randval(0.0, 1.0) < 0.6)
          seq_value = (1.0 - blend) * seq_value + blend * gbest[seq_idx];
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else
          seq_value += randval(-scale, scale);
#endif
        set_changed(seq_idx, seq_value);

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if MEME_GREEDY_CONFIGURATION_POLICY
        double greedy_device =
            duration + 0.38196601125 * operation_position +
            configuration_bias[configuration_action];
        greedy_device -= std::floor(greedy_device);
        double dev_value = current_value(dev_idx);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (configuration_action == 0) {
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (randval(0.0, 1.0) < 0.7)
            dev_value = 0.65 * dev_value + 0.35 * gbest[dev_idx];
          // 控制说明：条件不成立时执行互斥的备用处理路径。
          else
            dev_value += randval(-scale, scale);
        } else {
          dev_value = (1.0 - blend) * dev_value + blend * greedy_device;
        }
// 控制说明：选择当前编译配置对应的实现路径。
#else
        double dev_value = current_value(dev_idx);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (randval(0.0, 1.0) < 0.7)
          dev_value = (1.0 - blend) * dev_value + blend * gbest[dev_idx];
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else
          dev_value += randval(-scale, scale);
#endif
        set_changed(dev_idx, dev_value);
      }
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    for (int c = 0; c < (int)changed.size(); c++) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (values[c] > Ubound)
        values[c] = Ubound;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (values[c] < Lbound)
        values[c] = Lbound;
    }

    // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
    eval_sparse_meme_candidate(*this, popi, changed.data(), values.data(),
                               (int)changed.size(), true);
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::newpop_elite_line_search(int popi, int L, double scale) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  static vector<double> temp_storage;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (Nvar < 10000000 && (int)temp_storage.size() < Nvar)
    temp_storage.resize(Nvar);
  double* temp = Nvar < 10000000 ? temp_storage.data() : nullptr;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  int steps = L + 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (steps < 2)
    steps = 2;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (steps > 3)
    steps = 3;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int step = 0; step < steps; step++) {
    const double* target = (step % 2 == 0) ? gbest : ibest[popi];
    double blend_alpha = scale * (step + 1) / steps;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (blend_alpha > 1.0)
      blend_alpha = 1.0;

    // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
    int materialized_ready = 0;
    double temp_fit = 1e300;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (EvaluFunc == CED_Schedule_ParallelProxy) {
      temp_fit = CED_Schedule_ParallelProxy_DenseBlend(
          newpop[popi], const_cast<double*>(target), blend_alpha, temp,
          &materialized_ready, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
          CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
          AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad,
          EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET,
          CE_ST, CE_ET);
    } else {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Nvar; j++) {
        temp[j] = newpop[popi][j] + blend_alpha * (target[j] - newpop[popi][j]);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (temp[j] > Ubound)
          temp[j] = Ubound;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        else if (temp[j] < Lbound)
          temp[j] = Lbound;
      }
      materialized_ready = 1;
      temp_fit = EvaluFunc(
          temp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
          MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
          CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
          CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    if (temp_fit < newpop_fit[popi]) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (materialized_ready) {
        std::memcpy(newpop[popi], temp,
                    sizeof(double) * static_cast<size_t>(Nvar));
      } else {
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int j = 0; j < Nvar; j++) {
          newpop[popi][j] =
              newpop[popi][j] + blend_alpha * (target[j] - newpop[popi][j]);
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (newpop[popi][j] > Ubound)
            newpop[popi][j] = Ubound;
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          else if (newpop[popi][j] < Lbound)
            newpop[popi][j] = Lbound;
        }
      }
      newpop_fit[popi] = temp_fit;
    }
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::newpop_multi_scale_gaussian(int popi, int L, double scale) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  int dim_count = Nvar < 16 ? Nvar : 16;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(meme_gaussian_sigma) || meme_gaussian_sigma <= 0.0)
    meme_gaussian_sigma = scale > 1e-8 ? scale : 1e-3;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int trial_iter = 0; trial_iter < L; trial_iter++) {
    int changed[16];
    double values[16];
    int changed_count = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < dim_count; d++) {
      int bit = rand() % Nvar;
      bool seen = false;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int c = 0; c < changed_count; c++)
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (changed[c] == bit)
          seen = true;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (seen)
        continue;
      changed[changed_count] = bit;
      values[changed_count] =
          newpop[popi][bit] + randnorm(0.0, meme_gaussian_sigma);
      changed_count++;
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    const bool improved = eval_sparse_meme_candidate(
        *this, popi, changed, values, changed_count);
    meme_gaussian_trials++;
    const double eta =
        0.1 * pow(10.0 / (static_cast<double>(meme_gaussian_trials) + 10.0),
                  0.6);
    meme_gaussian_sigma *= exp(eta * ((improved ? 1.0 : 0.0) - 0.2));
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!std::isfinite(meme_gaussian_sigma) || meme_gaussian_sigma < 1e-12)
      meme_gaussian_sigma = 1e-12;
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::newpop_random_subspace_pattern(int popi, int L, double scale) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  double step = scale;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (step <= 1e-8)
    step = 1e-3;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int trial_iter = 0; trial_iter < L; trial_iter++) {
    int bit = rand() % Nvar;
    bool improved = false;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int direction = -1; direction <= 1; direction += 2) {
      int dims[1] = {bit};
      double values[1] = {newpop[popi][bit] + direction * step};
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (eval_sparse_meme_candidate(*this, popi, dims, values, 1)) {
        improved = true;
        break;
      }
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (improved)
      step *= 1.25;
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      step *= 0.5;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (step < 1e-6)
      step = 1e-6;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (step > scale * 4.0)
      step = scale * 4.0;
  }
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
bool eval_sparse_meme_candidate(MultiMet& solver, int popi, const int* dims,
                                const double* values, int count,
                                bool dimensions_are_unique) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (count <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return false;

  // Task-associated blocks are constructed from distinct tasks and therefore
  // contain distinct encoded dimensions.  Avoid rebuilding an identity map
  // and copying the candidate values when that property is known by the
  // caller.  This is mathematically identical to the generic path below.
  if (dimensions_are_unique) {
    vector<int> used;
    vector<double> old_values;
    used.reserve(count);
    old_values.reserve(count);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < count; ++i) {
      const int dim = dims[i];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (dim < 0 || dim >= solver.Nvar)
        continue;
      double value = values[i];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value > solver.Ubound)
        value = solver.Ubound;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (value < solver.Lbound)
        value = solver.Lbound;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value == solver.newpop[popi][dim])
        continue;
      used.push_back(dim);
      old_values.push_back(solver.newpop[popi][dim]);
      solver.newpop[popi][dim] = value;
    }
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (used.empty()) {
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return false;
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    const double fit = solver.EvaluFunc(
        solver.newpop[popi], solver.Cnum, solver.Enum, solver.Dnum,
        solver.CE_Tnum, solver.M_Jnum, solver.M_OPTnum,
        solver.CETask_Property, solver.MTask_Time, solver.EtoD_Distance,
        solver.DtoD_Distance, solver.AvailDeviceList, solver.EnergyList,
        solver.CloudDevices, solver.EdgeDevices, solver.CloudLoad,
        solver.EdgeLoad, solver.DeviceLoad, solver.CETask_coDevice,
        solver.Edge_Device_comm, solver.ST, solver.ET, solver.CE_ST,
        solver.CE_ET);
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (fit >= solver.newpop_fit[popi]) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (size_t i = 0; i < used.size(); ++i)
        solver.newpop[popi][used[i]] = old_values[i];
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return false;
    }
    solver.newpop_fit[popi] = fit;
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return true;
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  vector<int> used(count);
  vector<double> old_values(count);
  vector<double> new_values(count);
  int used_count = 0;
  std::unordered_map<int, int> large_index;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (!dimensions_are_unique && count > 64)
    large_index.reserve(static_cast<size_t>(count) * 2);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < count; i++) {
    int dim = dims[i];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (dim < 0 || dim >= solver.Nvar)
      continue;

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    double value = values[i];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (value > solver.Ubound)
      value = solver.Ubound;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (value < solver.Lbound)
      value = solver.Lbound;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (value == solver.newpop[popi][dim])
      continue;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    int found = -1;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (dimensions_are_unique) {
      found = -1;
    } else if (count > 64) {
      auto position = large_index.find(dim);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (position != large_index.end())
        found = position->second;
    } else {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int k = 0; k < used_count; k++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (used[k] == dim) {
          found = k;
          break;
        }
      }
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (found >= 0) {
      new_values[found] = value;
    } else {
      used[used_count] = dim;
      old_values[used_count] = solver.newpop[popi][dim];
      new_values[used_count] = value;
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (!dimensions_are_unique && count > 64)
        large_index.emplace(dim, used_count);
      used_count++;
    }
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (used_count <= 0) {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return false;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < used_count; i++)
    solver.newpop[popi][used[i]] = new_values[i];

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double fit = solver.EvaluFunc(
      solver.newpop[popi], solver.Cnum, solver.Enum, solver.Dnum,
      solver.CE_Tnum, solver.M_Jnum, solver.M_OPTnum, solver.CETask_Property,
      solver.MTask_Time, solver.EtoD_Distance, solver.DtoD_Distance,
      solver.AvailDeviceList, solver.EnergyList, solver.CloudDevices,
      solver.EdgeDevices, solver.CloudLoad, solver.EdgeLoad, solver.DeviceLoad,
      solver.CETask_coDevice, solver.Edge_Device_comm, solver.ST, solver.ET,
      solver.CE_ST, solver.CE_ET);
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (fit >= solver.newpop_fit[popi]) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < used_count; i++)
      solver.newpop[popi][used[i]] = old_values[i];
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return false;
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  solver.newpop_fit[popi] = fit;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return true;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
bool eval_sparse_meme_fit(MultiMet& solver, int popi, const int* dims,
                          const double* values, int count, double& fit) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (count <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return false;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  vector<int> used(count);
  vector<double> old_values(count);
  vector<double> new_values(count);
  int used_count = 0;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < count; i++) {
    int dim = dims[i];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (dim < 0 || dim >= solver.Nvar)
      continue;

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    double value = values[i];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (value > solver.Ubound)
      value = solver.Ubound;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (value < solver.Lbound)
      value = solver.Lbound;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (value == solver.newpop[popi][dim])
      continue;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    int found = -1;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k < used_count; k++) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (used[k] == dim) {
        found = k;
        break;
      }
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (found >= 0) {
      new_values[found] = value;
    } else {
      used[used_count] = dim;
      old_values[used_count] = solver.newpop[popi][dim];
      new_values[used_count] = value;
      used_count++;
    }
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (used_count <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return false;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < used_count; i++)
    solver.newpop[popi][used[i]] = new_values[i];

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  fit = solver.EvaluFunc(
      solver.newpop[popi], solver.Cnum, solver.Enum, solver.Dnum,
      solver.CE_Tnum, solver.M_Jnum, solver.M_OPTnum, solver.CETask_Property,
      solver.MTask_Time, solver.EtoD_Distance, solver.DtoD_Distance,
      solver.AvailDeviceList, solver.EnergyList, solver.CloudDevices,
      solver.EdgeDevices, solver.CloudLoad, solver.EdgeLoad, solver.DeviceLoad,
      solver.CETask_coDevice, solver.Edge_Device_comm, solver.ST, solver.ET,
      solver.CE_ST, solver.CE_ET);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < used_count; i++)
    solver.newpop[popi][used[i]] = old_values[i];
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return true;
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
// Ablation of the unified operator in the paper: every component is drawn
// independently for every candidate.  It intentionally does not select one
// of the eight pre-composed Meme structures.
struct ComponentContribution {
  double directional = 0.0;
  double disturbance = 0.0;
  double prior = 0.0;
  double direction[5] = {};
};

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void component_unified_candidate(MultiMet& solver, int popi, double scale,
                                 int selected_sampling = -1,
                                 int selected_direction = -1,
                                 const double* selected_direction_weights = nullptr,
                                 int selected_disturbance = -1,
                                 int selected_prior = -1,
                                 double selected_alpha = -1.0,
                                 double selected_beta = -1.0,
                                 double selected_gamma = -1.0,
                                 double selected_diversity = -1.0,
                                 double selected_gaussian_sigma = -1.0,
                                 ComponentContribution* contribution = nullptr) {
  const int n = solver.Nvar;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= solver.Popsize || n <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  const int sampling_rule = selected_sampling >= 0 ? selected_sampling % 4
                                                   : rand() % 4;
  const int prior_rule = selected_prior >= 0 ? selected_prior % 4 : rand() % 4;
  vector<int> dims;
  const bool policy_scaled_blocks = sampling_rule == 3 && selected_sampling >= 0;
  int structured_tasks_per_factory = 0;
  int structured_factory_count = 0;
  int structured_task_stride = 0;
  int structured_block_start = 0;
  int structured_blocks = 0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (sampling_rule == 0) {
    dims.push_back(rand() % n);
  } else if (sampling_rule == 1) {
    const int count = 1 + rand() % std::min(16, n);
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    while ((int)dims.size() < count) {
      const int dim = rand() % n;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (std::find(dims.begin(), dims.end(), dim) == dims.end())
        dims.push_back(dim);
    }
  } else if (sampling_rule == 2) {
    // Bernoulli sampling with a freshly drawn expected cardinality in [1,16].
    const double expected = solver.randval(1.0, std::min(16, n));
    const double probability = expected / n;
    const int forced = rand() % n;
    dims.push_back(forced);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (probability >= 1.0) {
      dims.resize(n);
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int d = 0; d < n; ++d)
        dims[d] = d;
    } else {
      const double log_miss = std::log1p(-probability);
      int d = 0;
      // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
      while (d < n) {
        const double u = std::max(1e-12, solver.randval(0.0, 1.0));
        d += static_cast<int>(std::log(u) / log_miss);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (d >= n)
          break;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (d != forced)
          dims.push_back(d);
        ++d;
      }
    }
  } else {
    // One or multiple complete task-associated blocks; their count is drawn
    // anew instead of being selected from the learned scope table.
    const int tasks_per_factory = std::min(1000, solver.CE_Tnum);
    const int factory_count = std::max(1, solver.CE_Tnum / tasks_per_factory);
    const long long block_work =
        static_cast<long long>(solver.M_Jnum) * solver.M_OPTnum;
    const int multi_factory_scope =
        block_work > 500000 ? 9 * tasks_per_factory / 10
                            : tasks_per_factory;
    const int blocks_per_factory = policy_scaled_blocks
        ? (solver.CE_Tnum <= 1000 ? 512 + rand() % 489
                                 : 1 + rand() % std::max(
                                       1, multi_factory_scope))
        : 1 + rand() % std::min(16, solver.CE_Tnum);
    const int blocks = policy_scaled_blocks
                           ? factory_count * blocks_per_factory
                           : blocks_per_factory;
    const int task_stride = policy_scaled_blocks
                                ? std::max(1, tasks_per_factory /
                                                  blocks_per_factory)
                                : 1;
    const int block_start = policy_scaled_blocks ? rand() % task_stride : 0;
    const int sequence_offset = 2 * solver.CE_Tnum;
    const int device_offset = sequence_offset +
                              solver.M_Jnum * solver.M_OPTnum;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (policy_scaled_blocks) {
      structured_tasks_per_factory = tasks_per_factory;
      structured_factory_count = factory_count;
      structured_task_stride = task_stride;
      structured_block_start = block_start;
      structured_blocks = blocks;
      // Preserve the original random stream: the former materialized-index
      // path drew and then overwrote one random task id for every block.
      for (int b = 0; b < blocks; ++b)
        (void)(rand() % solver.CE_Tnum);
    } else {
      dims.reserve(static_cast<size_t>(blocks) *
                   static_cast<size_t>(2 + 2 * solver.M_OPTnum));
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int b = 0; b < blocks; ++b) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (policy_scaled_blocks)
        break;
      int task = rand() % solver.CE_Tnum;
      dims.push_back(task);
      dims.push_back(solver.CE_Tnum + task);
      const int job = task % solver.M_Jnum;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int op = 0; op < solver.M_OPTnum; ++op) {
        const int operation = job * solver.M_OPTnum + op;
        dims.push_back(sequence_offset + operation);
        dims.push_back(device_offset + operation);
      }
    }
  }

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  const int disturbance_rule = selected_disturbance >= 0
                                   ? selected_disturbance % 5
                                   : rand() % 5;
  const int direction_count = selected_direction_weights != nullptr
                                  ? 5
                                  : (selected_direction >= 0 ? 1
                                                             : 1 + rand() % 5);
  int direction_ids[5] = {0, 1, 2, 3, 4};
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (selected_direction_weights != nullptr) {
    // The paper-defined policy combines all directional references through
    // learned normalized omega weights.
  } else if (selected_direction >= 0)
    direction_ids[0] = selected_direction % 5;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 4; k > 0; --k)
      std::swap(direction_ids[k], direction_ids[rand() % (k + 1)]);
  double direction_weights[5] = {};
  double weight_sum = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int k = 0; k < direction_count; ++k) {
    direction_weights[k] = selected_direction_weights != nullptr
                               ? std::max(0.0, selected_direction_weights[k])
                               : solver.randval(1e-12, 1.0);
    weight_sum += direction_weights[k];
  }
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (weight_sum <= 0.0)
    weight_sum = direction_count;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int k = 0; k < direction_count; ++k)
    direction_weights[k] /= weight_sum;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  const double alpha = selected_alpha >= 0.0 ? selected_alpha
                                             : solver.randval(0.0, 1.0);
  const double beta = selected_beta >= 0.0 ? selected_beta
                                           : solver.randval(0.0, 1.0);
  const double gamma = selected_gamma >= 0.0 ? selected_gamma
                                             : solver.randval(0.0, 1.0);
  const double radius_base = selected_diversity >= 0.0
                                 ? selected_diversity
                                 : scale;
  const int task_block_width = 2 + 2 * solver.M_OPTnum;
  const size_t dimension_count =
      policy_scaled_blocks
          ? static_cast<size_t>(structured_blocks) * task_block_width
          : dims.size();
  const double radius = radius_base /
                        std::sqrt((double)std::max<size_t>(1, dimension_count));
  static vector<double> reusable_structured_values;
  vector<double> transient_values;
  vector<double>& values =
      policy_scaled_blocks ? reusable_structured_values : transient_values;
  values.resize(dimension_count);
  const int sequence_offset = 2 * solver.CE_Tnum;
  const int operation_count = solver.M_Jnum * solver.M_OPTnum;
  const int device_offset = sequence_offset + operation_count;
  auto structured_dimension = [&](size_t position) {
    const int block = static_cast<int>(position / task_block_width);
    const int coordinate = static_cast<int>(position % task_block_width);
    const int factory = block % structured_factory_count;
    const int within_factory = block / structured_factory_count;
    int task = factory * structured_tasks_per_factory +
               structured_block_start +
               within_factory * structured_task_stride;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (task >= solver.CE_Tnum)
      task %= solver.CE_Tnum;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (coordinate == 0)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return task;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (coordinate == 1)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return solver.CE_Tnum + task;
    const int operation_in_job = (coordinate - 2) / 2;
    const int operation = (task % solver.M_Jnum) * solver.M_OPTnum +
                          operation_in_job;
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ((coordinate - 2) & 1) == 0 ? sequence_offset + operation
                                        : device_offset + operation;
  };
  auto for_each_structured_dimension = [&](auto&& operation) {
    size_t position = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int block = 0; block < structured_blocks; ++block) {
      const int factory = block % structured_factory_count;
      const int within_factory = block / structured_factory_count;
      int task = factory * structured_tasks_per_factory +
                 structured_block_start +
                 within_factory * structured_task_stride;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (task >= solver.CE_Tnum)
        task %= solver.CE_Tnum;
      operation(position++, task);
      operation(position++, solver.CE_Tnum + task);
      const int job = task % solver.M_Jnum;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int op = 0; op < solver.M_OPTnum; ++op) {
        const int encoded_operation = job * solver.M_OPTnum + op;
        operation(position++, sequence_offset + encoded_operation);
        operation(position++, device_offset + encoded_operation);
      }
    }
  };
  int block_coordinate_sign = 1;
  int block_history_sign = 1;
  double block_disturbance = 0.0;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  for (size_t p = 0; p < dimension_count; ++p) {
    const bool task_block_start = sampling_rule == 3 &&
                                  p % task_block_width == 0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (task_block_start) {
      block_coordinate_sign = (rand() & 1) ? 1 : -1;
      block_history_sign = (rand() & 1) ? 1 : -1;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (disturbance_rule == 0)
        block_disturbance = solver.randval(-radius, radius);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (disturbance_rule == 1)
        block_disturbance = solver.randval(0.0, radius);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (disturbance_rule == 2)
        block_disturbance = -solver.randval(0.0, radius);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (disturbance_rule == 3) {
        const double u = solver.randval(1e-6, 1.0 - 1e-6);
        block_disturbance = radius * std::tan(kPi * (u - 0.5));
        block_disturbance = meme_clamp_value(
            block_disturbance, -8.0 * radius, 8.0 * radius);
      } else {
        const double sigma = selected_gaussian_sigma >= 0.0
                                 ? selected_gaussian_sigma
                                 : solver.randval(1e-12,
                                                  std::max(1e-12, radius));
        block_disturbance = solver.randnorm(0.0, sigma);
      }
    }
    const int dim = policy_scaled_blocks ? structured_dimension(p) : dims[p];
    const double current = solver.newpop[popi][dim];
    double directional = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k < direction_count; ++k) {
      double delta = 0.0;
      // 控制说明：根据算法/动作枚举分派到对应实现，避免不同方法共享错误路径。
      switch (direction_ids[k]) {
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 0: delta = solver.gbest[dim] - current; break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 1: delta = solver.ibest[popi][dim] - current; break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 2:
        delta = (sampling_rule == 3 ? block_coordinate_sign
                                    : ((rand() & 1) ? 1 : -1)) *
                (solver.Ubound - solver.Lbound);
        break;
      // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
      case 3:
        delta = solver.Ubound + solver.Lbound - 2.0 * current;
        break;
      default: {
        const size_t index = ssals_step_index(solver, popi, dim);
        const double step = index < solver.ssals_step.size()
                                ? solver.ssals_step[index]
                                : ssals_initial_step(solver);
        delta = (sampling_rule == 3 ? block_history_sign
                                    : ((rand() & 1) ? 1 : -1)) * step;
        break;
      }
      }
      directional += direction_weights[k] * delta;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (contribution != nullptr)
        contribution->direction[direction_ids[k]] +=
            std::fabs(direction_weights[k] * delta);
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    double disturbance = sampling_rule == 3 ? block_disturbance : 0.0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (sampling_rule == 3) {
      // A task-associated block receives one coherent disturbance draw.
    } else if (disturbance_rule == 0)
      disturbance = solver.randval(-radius, radius);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (disturbance_rule == 1)
      disturbance = solver.randval(0.0, radius);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (disturbance_rule == 2)
      disturbance = -solver.randval(0.0, radius);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (disturbance_rule == 3) {
      const double u = solver.randval(1e-6, 1.0 - 1e-6);
      disturbance = radius * std::tan(kPi * (u - 0.5));
      disturbance = meme_clamp_value(disturbance, -8.0 * radius,
                                     8.0 * radius);
    } else {
      const double sigma = selected_gaussian_sigma >= 0.0
                               ? selected_gaussian_sigma
                               : solver.randval(1e-12,
                                                std::max(1e-12, radius));
      disturbance = solver.randnorm(0.0, sigma);
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    double prior = current;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (prior_rule != 0) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (dim < solver.CE_Tnum) {
        const int task = dim;
        const double comp = solver.CETask_Property[task].Computation / 199.0;
        const double comm = solver.CETask_Property[task].Communication / 4999.0;
        const double dep = (solver.CETask_Property[task].Precedence.size() +
                            solver.CETask_Property[task].Interact.size() +
                            solver.CETask_Property[task].Start_Pre.size() +
                            solver.CETask_Property[task].End_Pre.size()) / 8.0;
        // The implementation stores task placement B before resource key Z.
        prior = prior_rule == 1 ? 1.0 :
                (prior_rule == 2 ? (comp + dep <= comm + 0.5 ? 1.0 : 0.0)
                                 : 1.0 / (1.0 + std::exp(-3.0 *
                                             (comm + dep - comp))));
      } else if (dim < sequence_offset) {
        const int task = dim - solver.CE_Tnum;
        const double comp = solver.CETask_Property[task].Computation / 199.0;
        const double comm = solver.CETask_Property[task].Communication / 4999.0;
        const double dep = (solver.CETask_Property[task].Precedence.size() +
                            solver.CETask_Property[task].Interact.size() +
                            solver.CETask_Property[task].Start_Pre.size() +
                            solver.CETask_Property[task].End_Pre.size()) / 8.0;
        const double pos = (double)task / std::max(1, solver.CE_Tnum - 1);
        const double phase = prior_rule == 1 ? 0.173 :
                             (prior_rule == 2 ? 0.519 : 1.038);
        prior = prior_rule < 3 ? comm + 0.5 * comp - dep + 0.618 * pos + phase
                               : comp + 0.5 * comm + dep + 0.618 * pos + phase;
        prior -= std::floor(prior);
      } else {
        const bool sequence = dim < device_offset;
        const int operation = sequence ? dim - sequence_offset
                                       : dim - device_offset;
        const int op = operation % solver.M_OPTnum;
        const double duration = solver.MTask_Time[operation] / 300.0;
        const double position = (double)operation / std::max(1, operation_count - 1);
        const double stage = (double)op / std::max(1, solver.M_OPTnum - 1);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (sequence)
          prior = prior_rule == 1 ? duration :
                  (prior_rule == 2 ? 0.75 * stage + 0.25 * duration
                                   : duration + 0.5 * position + stage);
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else {
          const double phase = prior_rule == 1 ? 0.173 :
                               (prior_rule == 2 ? 0.519 : 1.038);
          prior = duration + 0.382 * position + phase;
        }
        prior -= std::floor(prior);
      }
    }

    // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
    values[p] = current + alpha * directional + beta * disturbance +
                gamma * (prior - current);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (contribution != nullptr) {
      contribution->directional += std::fabs(alpha * directional);
      contribution->disturbance += std::fabs(beta * disturbance);
      contribution->prior += std::fabs(gamma * (prior - current));
    }
  }
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (policy_scaled_blocks) {
    bool changed = false;
    for_each_structured_dimension([&](size_t p, int dim) {
      double candidate = values[p];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (candidate > solver.Ubound)
        candidate = solver.Ubound;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (candidate < solver.Lbound)
        candidate = solver.Lbound;
      const double old_value = solver.newpop[popi][dim];
      values[p] = old_value;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (candidate != old_value) {
        solver.newpop[popi][dim] = candidate;
        changed = true;
      }
    });
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!changed)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return;
    const double fit = solver.EvaluFunc(
        solver.newpop[popi], solver.Cnum, solver.Enum, solver.Dnum,
        solver.CE_Tnum, solver.M_Jnum, solver.M_OPTnum,
        solver.CETask_Property, solver.MTask_Time, solver.EtoD_Distance,
        solver.DtoD_Distance, solver.AvailDeviceList, solver.EnergyList,
        solver.CloudDevices, solver.EdgeDevices, solver.CloudLoad,
        solver.EdgeLoad, solver.DeviceLoad, solver.CETask_coDevice,
        solver.Edge_Device_comm, solver.ST, solver.ET, solver.CE_ST,
        solver.CE_ET);
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (fit < solver.newpop_fit[popi]) {
      solver.newpop_fit[popi] = fit;
    } else {
      for_each_structured_dimension(
          [&](size_t p, int dim) { solver.newpop[popi][dim] = values[p]; });
    }
  } else {
    eval_sparse_meme_candidate(solver, popi, dims.data(), values.data(),
                               (int)dims.size(), false);
  }
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct PaperComponentPolicyState {
  double sampling[MultiMet::HyperStateCount][4];
  double disturbance[MultiMet::HyperStateCount][5];
  double prior[MultiMet::HyperStateCount][4];
  double proxy[MultiMet::HyperStateCount][2];
  int sampling_trials[MultiMet::HyperStateCount][4];
  int disturbance_trials[MultiMet::HyperStateCount][5];
  int prior_trials[MultiMet::HyperStateCount][4];
  int proxy_trials[MultiMet::HyperStateCount][2];
  double alpha[MultiMet::HyperStateCount];
  double beta[MultiMet::HyperStateCount];
  double gamma[MultiMet::HyperStateCount];
  double omega[MultiMet::HyperStateCount][5];
  int continuous_trials[MultiMet::HyperStateCount];
  double gaussian_sigma[MultiMet::HyperStateCount];
  unsigned long gaussian_trials[MultiMet::HyperStateCount];
  const MultiMet* owner;
};

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
PaperComponentPolicyState& paper_component_policy(MultiMet& solver) {
  static PaperComponentPolicyState state = {};
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (state.owner == &solver)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return state;
  state = {};
  state.owner = &solver;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int h = 0; h < MultiMet::HyperStateCount; ++h) {
    std::fill_n(state.sampling[h], 4, 1.0);
    std::fill_n(state.disturbance[h], 5, 1.0);
    std::fill_n(state.prior[h], 4, 1.0);
    std::fill_n(state.proxy[h], 2, 1.0);
    // Uniformly initialize one of the seven nonempty component-activation
    // subsets.  Zero is a legitimate coefficient because several retained
    // operators omit direction, independent disturbance, or prior guidance.
    const int active_components = 1 + rand() % 7;
    state.alpha[h] = (active_components & 1)
                         ? solver.randval(0.0, 1.0) : 0.0;
    state.beta[h] = (active_components & 2)
                        ? solver.randval(0.0, 1.0) : 0.0;
    state.gamma[h] = (active_components & 4)
                         ? solver.randval(0.0, 1.0) : 0.0;
    state.gaussian_sigma[h] = -1.0;
    int direction_ids[5] = {0, 1, 2, 3, 4};
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 4; d > 0; --d)
      std::swap(direction_ids[d], direction_ids[rand() % (d + 1)]);
    const int active_directions = 1 + rand() % 5;
    double sum = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < active_directions; ++d) {
      state.omega[h][direction_ids[d]] = solver.randval(1e-12, 1.0);
      sum += state.omega[h][direction_ids[d]];
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < 5; ++d)
      state.omega[h][d] /= sum;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return state;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
int select_paper_action(MultiMet& solver, double* weights, int* trials,
                        int count) {
  const double epsilon = exp3_exploration_rate(trials, count);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return solver.choose_policy_action(weights, count, epsilon, false);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void update_paper_action(double* weights, int* trials, int action,
                         double reward) {
  ++trials[action];
  update_policy_cell(weights[action], trials[action], reward);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void update_paper_continuous(double& value, double eta, double reward) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (reward > 0.0)
    value += eta * reward;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    value *= 1.0 - eta / 4.0;
  value = meme_clamp_value(value, 0.0, 1.0);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
void run_paper_component_policy(MultiMet& solver, int gen, int max_gen,
                                double scale, int p_start, int p_end) {
  PaperComponentPolicyState& policy = paper_component_policy(solver);
  double progress = 0.0;
  const int h = solver.MemePolicyState(gen, max_gen, progress);
  const int c = 0; // continuous operator weights persist across all states
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (policy.gaussian_sigma[c] < 0.0)
    policy.gaussian_sigma[c] = scale;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; ++i) {
    const int sampling = select_paper_action(
        solver, policy.sampling[h], policy.sampling_trials[h], 4);
    const int disturbance = select_paper_action(
        solver, policy.disturbance[h], policy.disturbance_trials[h], 5);
    const int prior = select_paper_action(
        solver, policy.prior[h], policy.prior_trials[h], 4);
    const int proxy = select_paper_action(
        solver, policy.proxy[h], policy.proxy_trials[h], 2);


    // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
    const double old_fit = solver.newpop_fit[i];
    const int true_before = CED_ProxyTrueEvaluationCount();
    const clock_t work_start = clock();
    const long long audit_work =
        static_cast<long long>(solver.M_Jnum) * solver.M_OPTnum;
    const long long linear_audit_budget = 500000;
    const bool structured_prior_candidate = sampling == 3 && prior != 0;
    const bool reduce_true_checks =
        proxy == 0 || (audit_work > linear_audit_budget &&
                       !structured_prior_candidate);
    CED_SetProxyReduceTrueCheckHint(reduce_true_checks,
                                    policy.proxy[h][proxy]);
    CED_SetProxyDisagreementAuditHint(
        proxy == 1 && audit_work <= linear_audit_budget &&
        (sampling == 3 || prior != 0));
    ComponentContribution contribution;
    component_unified_candidate(
        solver, i, scale, sampling, -1, policy.omega[c], disturbance, prior,
        policy.alpha[c], policy.beta[c], policy.gamma[c], -1.0,
        policy.gaussian_sigma[c], &contribution);
    CED_ClearProxyPolicyHint();

    // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
    const int true_cost = CED_ProxyTrueEvaluationCount() - true_before;
    const double work_cost =
        static_cast<double>(clock() - work_start) / CLOCKS_PER_SEC;
    const double verified_gain = CED_LastProxyTrueRelativeImprovement();
    const double reward =
        true_cost > 0 && verified_gain >= 0.0
            ? sqrt(max(0.0, verified_gain)) /
                  sqrt(max(1e-12, work_cost) + true_cost)
            : meme_policy_reward_from_fit(old_fit, solver.newpop_fit[i],
                                          true_cost, work_cost);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (disturbance == 4) {
      ++policy.gaussian_trials[c];
      const double gaussian_eta = 0.1 * std::pow(
          10.0 / (policy.gaussian_trials[c] + 10.0), 0.6);
      const bool improved = solver.newpop_fit[i] < old_fit;
      policy.gaussian_sigma[c] *=
          std::exp(gaussian_eta * ((improved ? 1.0 : 0.0) - 0.2));
    }

    // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
    update_paper_action(policy.sampling[h], policy.sampling_trials[h],
                        sampling, reward);
    update_paper_action(policy.disturbance[h], policy.disturbance_trials[h],
                        disturbance, reward);
    update_paper_action(policy.prior[h], policy.prior_trials[h], prior,
                        reward);
    update_paper_action(policy.proxy[h], policy.proxy_trials[h], proxy,
                        reward);

    // 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
    ++policy.continuous_trials[c];
    const double eta = 0.1 * std::pow(
        10.0 / (policy.continuous_trials[c] + 10.0), 0.6);
    update_paper_continuous(policy.alpha[c], eta, reward);
    update_paper_continuous(policy.beta[c], eta, reward);
    update_paper_continuous(policy.gamma[c], eta, reward);
    double omega_sum = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < 5; ++d) {
      update_paper_continuous(policy.omega[c][d], eta, reward);
      omega_sum += policy.omega[c][d];
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (omega_sum <= 0.0)
      omega_sum = 5.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < 5; ++d)
      policy.omega[c][d] /= omega_sum;
  }
}
} // namespace

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::newpop_cauchy_basin_hop(int popi, int L, double scale) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  int trials = L + 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (trials < 1)
    trials = 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (trials > 3)
    trials = 3;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (Nvar > 500000)
    trials = 1;
  int dim_count = Nvar < 12 ? Nvar : 12;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int trial_iter = 0; trial_iter < trials; trial_iter++) {
    int changed[16];
    double values[16];
    int changed_count = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < dim_count; d++) {
      int bit = rand() % Nvar;
      bool seen = false;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int c = 0; c < changed_count; c++)
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (changed[c] == bit)
          seen = true;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (seen)
        continue;

      // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
      double u = randval(1e-6, 1.0 - 1e-6);
      double jump = scale * tan(kPi * (u - 0.5));
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (jump > scale * 8.0)
        jump = scale * 8.0;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (jump < -scale * 8.0)
        jump = -scale * 8.0;

      // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
      values[changed_count] = newpop[popi][bit] + jump;
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (values[changed_count] > Ubound)
        values[changed_count] = Ubound;
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      else if (values[changed_count] < Lbound)
        values[changed_count] = Lbound;
      changed[changed_count++] = bit;
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    double temp_fit = newpop_fit[popi];
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (!eval_sparse_meme_fit(*this, popi, changed, values, changed_count,
                              temp_fit))
      continue;
    double accept_prob = exp((newpop_fit[popi] - temp_fit) /
                             (fabs(newpop_fit[popi]) * 0.01 + 1e-12));
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (temp_fit < newpop_fit[popi] || randval(0.0, 1.0) < 0.05 * accept_prob) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int c = 0; c < changed_count; c++)
        newpop[popi][changed[c]] = values[c];
      newpop_fit[popi] = temp_fit;
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return;
    }
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::newpop_bit_climbing(int popi, int L, double scale) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int j = 0; j < L; j++) {
    int bit = rand() % Nvar;
    int dims[1] = {bit};
    double values[1] = {newpop[popi][bit] + scale * randval(Lbound, Ubound)};
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (values[0] > Ubound)
      values[0] = Ubound;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (values[0] < Lbound)
      values[0] = Lbound;

    // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
    eval_sparse_meme_candidate(*this, popi, dims, values, 1);
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::newpop_stepsize_adaptive(int popi, int L, double scale) {
  (void)scale;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if ((int)ssals_step.size() != Popsize * Nvar)
    ResetSSALSState();

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const double init_step = ssals_initial_step(*this);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int local_iter = 0; local_iter < L; local_iter++) {
    const double select_prob = 1.0 / static_cast<double>(Nvar);

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    const int forced_dim = rand() % Nvar;
    vector<int> dims;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (select_prob >= 1.0) {
      dims.reserve(Nvar);
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int dim = 0; dim < Nvar; dim++)
        dims.push_back(dim);
    } else if (select_prob > 0.01) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int dim = 0; dim < Nvar; dim++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (dim == forced_dim || randval(0.0, 1.0) < select_prob)
          dims.push_back(dim);
      }
    } else {
      dims.push_back(forced_dim);
      double log_miss = log(1.0 - select_prob);
      int dim = 0;
      // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
      while (dim < Nvar && log_miss < 0.0) {
        double u = randval(0.0, 1.0);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (u <= 0.0)
          u = 1e-12;
        int skip = (int)(log(u) / log_miss);
        dim += skip;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (dim >= Nvar)
          break;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (dim != forced_dim)
          dims.push_back(dim);
        dim++;
      }
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (dims.empty())
      dims.push_back(forced_dim);

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    vector<double> old_values(dims.size());
    vector<double> candidate_values(dims.size());
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (size_t i = 0; i < dims.size(); i++) {
      int dim = dims[i];
      size_t step_idx = ssals_step_index(*this, popi, dim);
      double step = ssals_step[step_idx];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (fabs(step) <= ssals_minbs || !std::isfinite(step)) {
        step = init_step;
        ssals_step[step_idx] = step;
      }

      // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
      old_values[i] = newpop[popi][dim];
      double value = old_values[i] + step * randval(0.0, 1.0);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value < Lbound || value > Ubound || !std::isfinite(value))
        value = randval(Lbound, Ubound);
      candidate_values[i] = value;
      newpop[popi][dim] = value;
    }

    // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
    const double old_fit = newpop_fit[popi];
    double candidate_fit =
        EvaluFunc(newpop[popi], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                  CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                  AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                  CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
                  Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    const bool improved = candidate_fit <= old_fit;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    if (improved) {
      newpop_fit[popi] = candidate_fit;
    } else {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (size_t i = 0; i < dims.size(); i++)
        newpop[popi][dims[i]] = old_values[i];
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    for (size_t i = 0; i < dims.size(); i++) {
      int dim = dims[i];
      size_t step_idx = ssals_step_index(*this, popi, dim);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (improved) {
        double effective_step = fabs(candidate_values[i] - old_values[i]);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (effective_step > 0.0)
          update_ssals_history(*this, dim, effective_step);
        ssals_step[step_idx] *= 2.0;
      } else {
        ssals_step[step_idx] *= -0.5;
      }

      // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
      if (fabs(ssals_step[step_idx]) <= ssals_minbs ||
          !std::isfinite(ssals_step[step_idx]))
        ssals_step[step_idx] = init_step;
    }
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::newpop_opposition_elite_blend(int popi, int L, double scale) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (popi < 0 || popi >= Popsize || L <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  static vector<double> temp_storage;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (Nvar < 10000000 && (int)temp_storage.size() < Nvar)
    temp_storage.resize(Nvar);
  double* temp = Nvar < 10000000 ? temp_storage.data() : nullptr;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  int trials = L + 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (trials < 2)
    trials = 2;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (trials > 3)
    trials = 3;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (Nvar > 500000)
    trials = 1;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int trial_iter = 0; trial_iter < trials; trial_iter++) {
    double opposition_alpha = randval(0.15, 0.85);
    double beta = randval(0.0, scale);
    const bool opposition = (trial_iter == 0);

    // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
    int materialized_ready = 0;
    double temp_fit = 1e300;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (EvaluFunc == CED_Schedule_ParallelProxy) {
      temp_fit = CED_Schedule_ParallelProxy_DenseOpposition(
          newpop[popi], gbest, ibest[popi], opposition_alpha, beta,
          opposition ? 1 : 0, temp, &materialized_ready, Cnum, Enum, Dnum,
          CE_Tnum, M_Jnum, M_OPTnum, CETask_Property, MTask_Time, EtoD_Distance,
          DtoD_Distance, AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
          CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm,
          ST, ET, CE_ST, CE_ET);
    } else {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Nvar; j++) {
        double opposite = Ubound + Lbound - newpop[popi][j];
        double elite = opposition_alpha * gbest[j] +
                       (1.0 - opposition_alpha) * ibest[popi][j];
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (opposition)
          temp[j] = 0.5 * opposite + 0.5 * elite;
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else
          temp[j] = newpop[popi][j] + beta * (elite - opposite);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (temp[j] > Ubound)
          temp[j] = Ubound;
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        else if (temp[j] < Lbound)
          temp[j] = Lbound;
      }
      materialized_ready = 1;
      temp_fit = EvaluFunc(
          temp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
          MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
          CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
          CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    if (temp_fit < newpop_fit[popi]) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (materialized_ready) {
        std::memcpy(newpop[popi], temp,
                    sizeof(double) * static_cast<size_t>(Nvar));
      } else {
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int j = 0; j < Nvar; j++) {
          double opposite = Ubound + Lbound - newpop[popi][j];
          double elite = opposition_alpha * gbest[j] +
                         (1.0 - opposition_alpha) * ibest[popi][j];
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (opposition)
            newpop[popi][j] = 0.5 * opposite + 0.5 * elite;
          // 控制说明：条件不成立时执行互斥的备用处理路径。
          else
            newpop[popi][j] = newpop[popi][j] + beta * (elite - opposite);
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          if (newpop[popi][j] > Ubound)
            newpop[popi][j] = Ubound;
          // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
          else if (newpop[popi][j] < Lbound)
            newpop[popi][j] = Lbound;
        }
      }
      newpop_fit[popi] = temp_fit;
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return;
    }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::meme_selection(int popi, int meme_id, double scale,
                              int iterations) {
  // 控制说明：根据算法/动作枚举分派到对应实现，避免不同方法共享错误路径。
  switch (meme_id) {
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 0: {
    newpop_lightweight_meme(popi, iterations, scale);
    break;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 1: {
    newpop_bit_climbing(popi, iterations, scale);
    break;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 2: {
    newpop_elite_line_search(popi, iterations, scale);
    break;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 3: {
    newpop_multi_scale_gaussian(popi, iterations, scale);
    break;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 4: {
    newpop_random_subspace_pattern(popi, iterations, scale);
    break;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 5: {
    newpop_opposition_elite_blend(popi, iterations, scale);
    break;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 6: {
    newpop_cauchy_basin_hop(popi, iterations, scale);
    break;
  }
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 7: {
    newpop_stepsize_adaptive(popi, iterations, scale);
    break;
  }
  default: {
    newpop_lightweight_meme(popi, iterations, scale);
    break;
  }
  }
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
void MultiMet::ResetMemePolicyState() {
  meme_gaussian_sigma = 0.0;
  meme_gaussian_trials = 0;
  forced_meme_action = -1;
  current_meme_configuration = 0;
  current_meme_scope = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int state = 0; state < HyperStateCount; state++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemeConfigurationActionCount; action++) {
      configuration_policy_trials[state][action] = 0;
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemeScopeActionCount; action++) {
      scope_policy_trials[state][action] = 0;
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemePolicyActionCount; action++) {
      meme_policy_trials[state][action] = 0;
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemeProxyPolicyActionCount; action++) {
      proxy_policy_trials[state][action] = 0;
    }
  }
  std::fill_n(meme_verified_successes, MemePolicyActionCount, 0);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
int MultiMet::MemePolicyState(int gen, int max_gen, double& progress) {
  progress = max_gen > 0 ? (double)gen / (double)max_gen : 1.0;
  progress = meme_clamp_value(progress, 0.0, 1.0);
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (!std::isfinite(meme_last_best) || meme_last_best >= 1e299) {
    meme_last_best = gbest_fit;
    meme_stagnant_generations = 0;
  } else if (gbest_fit < meme_last_best) {
    meme_last_best = gbest_fit;
    meme_stagnant_generations = 0;
  } else {
    meme_stagnant_generations++;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return search_state(progress, meme_stagnant_generations);
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
int MultiMet::SelectPolicyMemeAction(int hyper_state, double progress) {
  hyper_state = (int)meme_clamp_value(hyper_state, 0, HyperStateCount - 1);
  (void)progress;
  const double epsilon = exp3_exploration_rate(meme_policy_trials[hyper_state],
                                               kMemeOperatorCount);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return choose_policy_action(meme_policy.meme[hyper_state], kMemeOperatorCount,
                              epsilon, kMemePolicySquaredWeight);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
int MultiMet::SelectPolicyProxyAction(int hyper_state, double progress) {
  hyper_state = (int)meme_clamp_value(hyper_state, 0, HyperStateCount - 1);
  (void)progress;
  const double epsilon = exp3_exploration_rate(proxy_policy_trials[hyper_state],
                                               MemeProxyPolicyActionCount);
  const int learned_action =
      choose_policy_action(meme_policy.proxy[hyper_state],
                           MemeProxyPolicyActionCount, epsilon,
                           kMemePolicySquaredWeight);
// 控制说明：选择当前编译配置对应的实现路径。
#if defined(PROXY_POLICY_ABLATION) && PROXY_POLICY_ABLATION == 1
  (void)learned_action;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 0;
#elif defined(PROXY_POLICY_ABLATION) && PROXY_POLICY_ABLATION == 2
  (void)learned_action;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 1;
// 控制说明：选择当前编译配置对应的实现路径。
#else
  return learned_action;
#endif
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::RunMemeSearchStrategy(int mode, int Gen, int MaxG, double scale,
                                     int p_start, int p_end) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (mode < 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (p_start < 0)
    p_start = 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (p_end > Popsize)
    p_end = Popsize;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (p_start >= p_end)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  switch (mode) {
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 1:
    meme_random_walk(scale, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 2:
    meme_simple_random(scale, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 3:
    meme_randperm(scale, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 4:
    meme_inheritance(scale, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case 5:
    meme_subprob_decomposition(Gen, MaxG, 4, scale, p_start, p_end);
    break;
  default:
    meme_policy_guidance(Gen, MaxG, scale, p_start, p_end);
    break;
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::meme_random_walk(double scale, int p_start, int p_end) {
  int step = 0, count = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < p_end; i++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (i >= p_start)
      meme_selection(i, step % kMemeOperatorCount, scale, 10);

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    double rr = randval(0, 1);
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (count < 5) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (rr < 0.5)
        step -= 1;
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        step += 1;
    } else {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (rr < 0.2)
        step -= 1;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (rr < 0.4)
        step += 1;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (rr < 0.55)
        step -= 2;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (rr < 0.7)
        step += 2;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (rr < 0.85)
        step -= 3;
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        step += 3;
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (step < 0)
      step = 0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (step > 4)
      step %= 4;
    count++;
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::meme_simple_random(double scale, int p_start, int p_end) {
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++)
    meme_selection(i, RandomMemeOperatorId(), scale, 10);
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::meme_randperm(double scale, int p_start, int p_end) {
  std::random_device rd;
  std::mt19937_64 g(rd());
  vector<int> permu;
  const int perm_count = 4;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < perm_count; i++)
    permu.push_back(i);
  shuffle(permu.begin(), permu.end(), g);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  int count = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < p_end; i++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (i >= p_start)
      meme_selection(i, permu[count], scale, 10);
    count++;
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (count >= perm_count) {
      count = 0;
      shuffle(permu.begin(), permu.end(), g);
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::meme_inheritance(double scale, int p_start, int p_end) {
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++)
    meme_selection(i, Ind_meme[i], scale, 10);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  int pair_start = p_start;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (pair_start & 1)
    pair_start--;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = pair_start; i < p_end - 1; i += 2) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (i < p_start || i + 1 >= p_end)
      continue;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (newpop_fit[i] == newpop_fit[i + 1]) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (randval(0, 1) < 0.5)
        Ind_meme[i] = Ind_meme[i + 1];
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        Ind_meme[i + 1] = Ind_meme[i];
    } else if (newpop_fit[i] < newpop_fit[i + 1])
      Ind_meme[i + 1] = Ind_meme[i];
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      Ind_meme[i] = Ind_meme[i + 1];
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::meme_subprob_decomposition(int Gen, int MaxG, int kk,
                                          double scale, int p_start,
                                          int p_end) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (kk < 1)
    kk = 1;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (kk > Popsize)
    kk = Popsize;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  const int sample_dim = (Nvar < 64) ? Nvar : 64;
  vector<double> distance(Popsize, 0.0);
  vector<int> order(Popsize, 0);

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  if (Gen < MaxG) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = p_start; i < p_end; i++) {
      int meme_id = RandomMemeOperatorId();
      double oldfit = newpop_fit[i];
      meme_selection(i, meme_id, scale, 10);
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Nvar; j++)
        SubDecBase[i][j] = newpop[i][j];
      SubDecBase[i][Nvar] = (gbest_fit / (newpop_fit[i] + 1e-12)) *
                            (oldfit - newpop_fit[i]) / 100.0;
      SubDecBase[i][Nvar + 1] = meme_id;
    }
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Popsize; j++) {
      order[j] = j;
      distance[j] = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int d = 0; d < sample_dim; d++) {
        int dim = (d * 9973 + i * 101 + j * 17) % Nvar;
        double diff = newpop[i][dim] - SubDecBase[j][dim];
        distance[j] += diff * diff;
      }
    }
    sort(order.begin(), order.end(),
         [&](int a, int b) { return distance[a] < distance[b]; });

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    int best = order[0];
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 1; k < kk; k++)
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (SubDecBase[order[k]][Nvar] > SubDecBase[best][Nvar])
        best = order[k];

    // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
    int meme_id = (int)SubDecBase[best][Nvar + 1];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!IsRetainedMemeOperator(meme_id))
      meme_id = RandomMemeOperatorId();
    Ind_meme[i] = meme_id;
    double oldfit = newpop_fit[i];
    meme_selection(i, meme_id, scale, 10);
    double reward = (gbest_fit / (newpop_fit[i] + 1e-12)) *
                    (oldfit - newpop_fit[i]) / 100.0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (reward > SubDecBase[i][Nvar]) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Nvar; j++)
        SubDecBase[i][j] = newpop[i][j];
      SubDecBase[i][Nvar] = reward;
      SubDecBase[i][Nvar + 1] = meme_id;
    }
  }
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::meme_policy_guidance(int Gen, int MaxG, double scale,
                                    int p_start, int p_end) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (p_start < 0)
    p_start = 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (p_end > Popsize)
    p_end = Popsize;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (p_start >= p_end)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if PAPER_COMPONENT_POLICY
  run_paper_component_policy(*this, Gen, MaxG, scale, p_start, p_end);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return;
#endif

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  double progress = 1.0;
  int hyper_state = MemePolicyState(Gen, MaxG, progress);
  double best_after = meme_last_best;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    const bool policy_ready = has_sufficient_policy_evidence(*this);
    const int meme_action =
        forced_meme_action >= 0
            ? forced_meme_action
// 控制说明：选择当前编译配置对应的实现路径。
#if TRI_POLICY_RANDOM_ABLATION
            : [&]() {
                double uniform_weights[MemePolicyActionCount];
                std::fill_n(uniform_weights, MemePolicyActionCount, 1.0);
                // 控制说明：返回本阶段计算结果或状态码给调用方。
                return choose_policy_action(uniform_weights,
                                            MemePolicyActionCount, 1.0,
                                            false);
              }();
// 控制说明：选择当前编译配置对应的实现路径。
#else
            : (policy_ready
                   ? SelectPolicyMemeAction(hyper_state, progress)
                   : choose_uniform_action(*this, MemePolicyActionCount));
#endif
    forced_meme_action = -1;
    const int meme_id = meme_action;
// 控制说明：选择当前编译配置对应的实现路径。
#if MEME_GREEDY_CONFIGURATION_POLICY
    int configuration_action = 0;
    int scope_action = 0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (meme_id == 0) {
// 控制说明：选择当前编译配置对应的实现路径。
#if TRI_POLICY_RANDOM_ABLATION
      double uniform_configuration[MemeConfigurationActionCount];
      double uniform_scope[MemeScopeActionCount];
      std::fill_n(uniform_configuration, MemeConfigurationActionCount, 1.0);
      std::fill_n(uniform_scope, MemeScopeActionCount, 1.0);
      configuration_action = choose_policy_action(
          uniform_configuration, MemeConfigurationActionCount, 1.0, false);
      scope_action = choose_policy_action(uniform_scope, MemeScopeActionCount,
                                          1.0, false);
// 控制说明：选择当前编译配置对应的实现路径。
#else
      if (!policy_ready) {
        configuration_action =
            choose_uniform_action(*this, MemeConfigurationActionCount);
      } else {
        const double configuration_epsilon = exp3_exploration_rate(
            configuration_policy_trials[hyper_state],
            MemeConfigurationActionCount);
        configuration_action = choose_policy_action(
            meme_policy.configuration[hyper_state],
            MemeConfigurationActionCount, configuration_epsilon,
            kMemePolicySquaredWeight);
      }
      current_meme_configuration = configuration_action;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (!policy_ready) {
        scope_action = choose_uniform_action(*this, MemeScopeActionCount);
      } else {
        const double scope_epsilon = exp3_exploration_rate(
            scope_policy_trials[hyper_state], MemeScopeActionCount);
        scope_action = choose_policy_action(
            meme_policy.scope[hyper_state], MemeScopeActionCount,
            scope_epsilon, kMemePolicySquaredWeight);
      }
#endif
#if FORCE_GREEDY_CONFIGURATION_ACTION >= 0
      configuration_action = FORCE_GREEDY_CONFIGURATION_ACTION %
                             MemeConfigurationActionCount;
#endif
#if FORCE_GREEDY_SCOPE_ACTION >= 0
      scope_action = FORCE_GREEDY_SCOPE_ACTION % MemeScopeActionCount;
#endif
      current_meme_scope = scope_action;
      current_meme_configuration = configuration_action;
    }
// 控制说明：选择当前编译配置对应的实现路径。
#else
    const int configuration_action = 0;
    const int scope_action = 0;
#endif
#if TRI_POLICY_RANDOM_ABLATION
    double uniform_proxy[MemeProxyPolicyActionCount];
    std::fill_n(uniform_proxy, MemeProxyPolicyActionCount, 1.0);
    const int proxy_action = choose_policy_action(
        uniform_proxy, MemeProxyPolicyActionCount, 1.0, false);
// 控制说明：选择当前编译配置对应的实现路径。
#else
    const int proxy_action =
        policy_ready ? SelectPolicyProxyAction(hyper_state, progress)
                     : choose_uniform_action(*this,
                                             MemeProxyPolicyActionCount);
#endif
    const double old_fit = newpop_fit[i];
    const int true_before = CED_ProxyTrueEvaluationCount();
    const clock_t work_start = clock();

    // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
    Ind_meme[i] = meme_id;
    CED_SetProxyReduceTrueCheckHint(
        proxy_action == 0,
        policy_ready ? meme_policy.proxy[hyper_state][proxy_action] : 1.0);
    double best_meme_weight = meme_policy.meme[hyper_state][0];
    double worst_meme_weight = meme_policy.meme[hyper_state][0];
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int action = 0; action < MemePolicyActionCount; ++action) {
      best_meme_weight =
          std::max(best_meme_weight, meme_policy.meme[hyper_state][action]);
      worst_meme_weight =
          std::min(worst_meme_weight, meme_policy.meme[hyper_state][action]);
    }
    const double audit_exploration = exp3_exploration_rate(
        meme_policy_trials[hyper_state], MemePolicyActionCount);
    const double audit_weight_floor =
        best_meme_weight -
        audit_exploration * (best_meme_weight - worst_meme_weight);
    const bool policy_supported_meme =
        !policy_ready ||
        meme_policy.meme[hyper_state][meme_action] >= audit_weight_floor;
    CED_SetProxyDisagreementAuditHint(proxy_action == 1 &&
                                      policy_supported_meme);
// 控制说明：选择当前编译配置对应的实现路径。
#if TRI_POLICY_FULL_COMPONENT_RANDOM_ABLATION
    component_unified_candidate(*this, i, scale);
// 控制说明：选择当前编译配置对应的实现路径。
#else
    meme_selection(i, meme_id, scale, kMemeIter);
#endif
    CED_ClearProxyPolicyHint();

    // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
    const int true_cost = CED_ProxyTrueEvaluationCount() - true_before;
    const double work_cost =
        static_cast<double>(clock() - work_start) / CLOCKS_PER_SEC;
    const double verified_gain = CED_LastProxyTrueRelativeImprovement();
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (true_cost > 0 && verified_gain > 0.0)
      meme_verified_successes[meme_action]++;
    const double reward =
        true_cost > 0 && verified_gain >= 0.0
            ? sqrt(max(0.0, verified_gain)) /
                  sqrt(max(1e-12, work_cost) + true_cost)
            : meme_policy_reward_from_fit(old_fit, newpop_fit[i], true_cost,
                                          work_cost);
// 控制说明：选择当前编译配置对应的实现路径。
#if !TRI_POLICY_RANDOM_ABLATION
    int& meme_trials = meme_policy_trials[hyper_state][meme_action];
    meme_trials++;
    update_policy_cell(meme_policy.meme[hyper_state][meme_action], meme_trials,
                       reward);
// 控制说明：选择当前编译配置对应的实现路径。
#if MEME_GREEDY_CONFIGURATION_POLICY
    if (meme_id == 0) {
      int& configuration_trials =
          configuration_policy_trials[hyper_state][configuration_action];
      configuration_trials++;
      update_policy_cell(
          meme_policy.configuration[hyper_state][configuration_action],
          configuration_trials, reward);
      int& scope_trials = scope_policy_trials[hyper_state][scope_action];
      scope_trials++;
      update_policy_cell(meme_policy.scope[hyper_state][scope_action],
                         scope_trials, reward);
    }
#endif
    int& proxy_trials = proxy_policy_trials[hyper_state][proxy_action];
    proxy_trials++;
    update_policy_cell(meme_policy.proxy[hyper_state][proxy_action],
                       proxy_trials, reward);
#endif
    if (newpop_fit[i] < best_after)
      best_after = newpop_fit[i];
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (best_after < meme_last_best) {
    meme_last_best = best_after;
    meme_stagnant_generations = 0;
  }
}
