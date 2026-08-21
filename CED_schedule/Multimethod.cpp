// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include "Multimethod.h"
#include "Config.h"
#ifndef DATA_FILE_PATH
#if defined(TNUM) && TNUM >= 1000000
#define DATA_FILE_PATH "data/datamatrix_1000000_compact"
#elif defined(TNUM) && TNUM >= 100000
#define DATA_FILE_PATH "data/datamatrix_100000"
#elif defined(TNUM) && TNUM >= 10000
#define DATA_FILE_PATH "data/datamatrix_10000"
#elif defined(TNUM) && TNUM >= 1000
#define DATA_FILE_PATH "data/datamatrix_1000"
// 控制说明：选择当前编译配置对应的实现路径。
#else
#define DATA_FILE_PATH                                                         \
  "/Users/lailiyuanjun/Desktop/data_generator/data_matrix_100.txt"
#endif
#endif
#ifndef POWER_FILE_PATH
#define POWER_FILE_PATH "data/Power_Consumption.txt"
#endif
#ifndef ADE_SKIP_EVAL_IMMIGRANTS
#define ADE_SKIP_EVAL_IMMIGRANTS 1
#endif
static DistanceValue EncodeDistance(double value) {
  double scaled = value / DISTANCE_SCALE;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (scaled <= 0.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (scaled >= 65535.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 65535;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return static_cast<DistanceValue>(scaled + 0.5);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
static DistanceValue** CreateDistanceMatrix(int nRow, int nCol) {
  DistanceValue** matrix = new DistanceValue*[nRow];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < nRow; i++)
    matrix[i] = new DistanceValue[nCol];
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return matrix;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
static void DeleteDistanceMatrix(DistanceValue** matrix, int nRow) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (matrix == nullptr)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < nRow; i++)
    delete[] matrix[i];
  delete[] matrix;
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
constexpr int kMaxOaDim = 2048;

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static void ReportDataReadFailure(const char* section, int outer_index = -1,
                                  int inner_index = -1) {
  cerr << "Invalid or incomplete data file: " << DATA_FILE_PATH
       << " while reading " << section;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (outer_index >= 0)
    cerr << " at index " << outer_index;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (inner_index >= 0)
    cerr << ", " << inner_index;
  cerr << ". Check whether TNUM/ENUM/DNUM/MOPT_NUM match the generated data."
       << endl;
  exit(1);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
template <typename T>
static void ReadDataValue(fstream& fs, T& value, const char* section,
                          int outer_index = -1, int inner_index = -1) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!(fs >> value))
    ReportDataReadFailure(section, outer_index, inner_index);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
template <typename T>
static void ReadPowerValue(fstream& fs, T& value, const char* section,
                           int index = -1) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!(fs >> value)) {
    cerr << "Invalid or incomplete power file: " << POWER_FILE_PATH
         << " while reading " << section;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (index >= 0)
      cerr << " at index " << index;
    cerr << "." << endl;
    exit(1);
  }
}

// 段落说明：实现 `particle_subspace_dim`：完成该函数负责的数据准备、算法步骤和状态返回。
int particle_subspace_dim(int nvar) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return (nvar < 64) ? nvar : 64;
}

// 段落说明：实现 `particle_dim_index`：完成该函数负责的数据准备、算法步骤和状态返回。
int particle_dim_index(int nvar, int particle, int step, int salt) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return (step * 9973 + particle * 101 + salt * 4099) % nvar;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
double bounded_particle_value(double value, double low, double high) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value > high)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return low;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value < low)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return high;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return value;
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
double clamp_particle_value(double value, double low, double high) {
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
} // namespace

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
double MultiMet::randnorm(double miu, double score) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return miu + score * sqrt(abs(-2 * log((rand() + 1) / (RAND_MAX + 1.0)))) *
                   cos(2 * kPi * rand() / (RAND_MAX + 1.0)); // by lyl
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
int MultiMet::search_state(double progress, int stagnant_generations) {
  const int stagnation_threshold = progress < 0.30 ? 80 : 50;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (stagnant_generations >= stagnation_threshold)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return HyperStagnation;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (progress < 0.30)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return HyperEarly;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (progress < 0.75)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return HyperMiddle;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return HyperLate;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
int MultiMet::choose_policy_action(const double* policy, int count,
                                   double epsilon, bool squared_weight) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (count <= 1)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0;
  epsilon = max(0.0, min(1.0, epsilon));

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  double sum = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < count; i++) {
    double weight = policy[i] > 1e-6 ? policy[i] : 1e-6;
    sum += squared_weight ? weight * weight : weight;
  }

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  const double safe_sum = max(sum, 1e-12);
  double pick = randval(0.0, 1.0);
  double cumulative = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < count; i++) {
    double weight = policy[i] > 1e-6 ? policy[i] : 1e-6;
    const double exploitation =
        (squared_weight ? weight * weight : weight) / safe_sum;
    cumulative += epsilon / count + (1.0 - epsilon) * exploitation;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (pick <= cumulative)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return i;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return count - 1;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::pop_update(int p_start, int p_end) {
  const size_t bytes = sizeof(double) * static_cast<size_t>(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    std::memcpy(pop[i], newpop[i], bytes);
    pop_fit[i] = newpop_fit[i];

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (newpop_fit[i] < ibest_fit[i]) {
      std::memcpy(ibest[i], newpop[i], bytes);
      ibest_fit[i] = newpop_fit[i];
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::pop_better_update(int p_start, int p_end) {
  const size_t bytes = sizeof(double) * static_cast<size_t>(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (newpop_fit[i] < pop_fit[i]) {
      std::memcpy(pop[i], newpop[i], bytes);
      pop_fit[i] = newpop_fit[i];
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (newpop_fit[i] < ibest_fit[i]) {
      std::memcpy(ibest[i], newpop[i], bytes);
      ibest_fit[i] = newpop_fit[i];
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::meme_state_update(int p_start, int p_end) {
  const size_t bytes = sizeof(double) * static_cast<size_t>(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    pop_fit[i] = newpop_fit[i];

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (newpop_fit[i] < ibest_fit[i]) {
      std::memcpy(ibest[i], newpop[i], bytes);
      ibest_fit[i] = newpop_fit[i];
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (newpop_fit[i] < gbest_fit) {
      std::memcpy(gbest, newpop[i], bytes);
      gbest_fit = newpop_fit[i];
    }
  }
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
static const SearchAlg kPreferredSearchAlgorithms[] = {
    SearchAlg::ADE,   SearchAlg::HS,    SearchAlg::NGA,   SearchAlg::VAGA,
    SearchAlg::EAGA,  SearchAlg::DE,    SearchAlg::GA,    SearchAlg::CSA,
    SearchAlg::ACO,   SearchAlg::PSO,   SearchAlg::CPSO,  SearchAlg::CMPSO,
    SearchAlg::APSO1, SearchAlg::APSO2, SearchAlg::APSO3, SearchAlg::APSO4,
    SearchAlg::POA};

// 段落说明：实现 `PreferredSearchAlgorithmCount`：完成该函数负责的数据准备、算法步骤和状态返回。
static int PreferredSearchAlgorithmCount() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return (int)(sizeof(kPreferredSearchAlgorithms) /
               sizeof(kPreferredSearchAlgorithms[0]));
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
SearchAlg MultiMet::ConfiguredSearchAlg(int configured_alg) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (configured_alg < (int)SearchAlg::Adaptive ||
      configured_alg > (int)SearchAlg::BA)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return SearchAlg::Adaptive;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return (SearchAlg)configured_alg;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::ResetAdaptiveGlobalSearch() {
  int alg_count = PreferredSearchAlgorithmCount();
  SearchCount.assign(alg_count, 0);
  SearchReward.assign(alg_count, 0.0);
  SearchRecentReward.assign(alg_count, 0.0);
  SearchStaleCount.assign(alg_count, 0);
  SearchTotal = 0;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
static int PreferredSearchAlgorithmIndex(SearchAlg algorithm) {
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < PreferredSearchAlgorithmCount(); i++)
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (kPreferredSearchAlgorithms[i] == algorithm)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return i;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 0;
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
SearchAlg MultiMet::SelectAdaptiveGlobalAlgorithm() {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (SearchCount.empty() || SearchReward.empty() || SearchRecentReward.empty())
    ResetAdaptiveGlobalSearch();

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  int alg_count = PreferredSearchAlgorithmCount();
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < alg_count; i++)
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (SearchCount[i] == 0)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return kPreferredSearchAlgorithms[i];

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  int best_index = 0;
  double best_score = -1.0;
  const double total_log = log((double)max(1, SearchTotal));
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < alg_count; i++) {
    const double average_reward = SearchReward[i] / SearchCount[i];
    const double confidence =
        sqrt(2.0 * total_log / static_cast<double>(SearchCount[i]));
    const double score = average_reward + confidence;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (score > best_score) {
      best_score = score;
      best_index = i;
    }
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  return kPreferredSearchAlgorithms[best_index];
}

// 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
void MultiMet::UpdateAdaptiveGlobalAlgorithm(SearchAlg algorithm,
                                             double reward) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (SearchCount.empty() || SearchReward.empty() || SearchRecentReward.empty())
    ResetAdaptiveGlobalSearch();

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  int alg_index = PreferredSearchAlgorithmIndex(algorithm);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!(reward > 0.0))
    reward = 0.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (reward > 1.0)
    reward = 1.0;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  SearchCount[alg_index]++;
  SearchReward[alg_index] += reward;
  SearchRecentReward[alg_index] =
      SearchReward[alg_index] / SearchCount[alg_index];
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (reward > 1e-8)
    SearchStaleCount[alg_index] = 0;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    SearchStaleCount[alg_index]++;
  SearchTotal++;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::RunSelectedGlobalSearch(SearchAlg algorithm, int generation,
                                       int max_generation, int p_start,
                                       int p_end, bool single_individual_mode) {
  // 控制说明：根据算法/动作枚举分派到对应实现，避免不同方法共享错误路径。
  switch (algorithm) {
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::GA:
    GA(0.8, 0.15, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::NGA:
    NGA(0.8, 0.15, 0.05, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::LGA:
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (single_individual_mode)
      GA(0.8, 0.15, p_start, p_end);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      LGA(0.8, 0.15, 0.05, 1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::VAGA:
    VAGA(p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::EAGA:
    EAGA(p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::PSO:
    PSO(0.8, 2.0, 2.0, 0.1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::CPSO:
    CPSO(0.8, 2.0, 2.0, 0.1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::SPSO:
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (single_individual_mode)
      PSO(0.8, 2.0, 2.0, 0.1, p_start, p_end);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      SPSO(0.8, 2.0, 2.0, 0.1, generation, max_generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::CMPSO:
    CMPSO(0.8, 2.0, 2.0, 0.1, generation, max_generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::APSO1:
    APSO_1(2.0, 2.0, 0.1, generation, max_generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::APSO2:
    APSO_2(2.0, 2.0, 0.1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::APSO3:
    APSO_3(0.1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::APSO4:
    APSO_4(0.1, generation, max_generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::APSO5:
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (single_individual_mode)
      APSO_4(0.1, generation, max_generation, p_start, p_end);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      APSO_5(0.1, generation, max_generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::OLPSO:
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (single_individual_mode)
      PSO(0.8, 2.0, 2.0, 0.1, p_start, p_end);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      OLPSO(0.8, 2.0, 0.1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::HS:
    HS(0.9, 0.3, 0.001, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::ACO:
    ACO(0.85, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::COA:
    COA(10, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::DE:
    DE(0.5, 1, 0.5, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::CSA:
    CSA(0.3, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::ABCA:
    ABCA(100, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::POA:
    POA(p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::ILS:
    ILS(1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::VNS:
    VNS(1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::GRASP:
    GRASP(0.3, 1, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::PBILC:
    PBILC(p_start, p_end, 0.1);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::BATA:
    BATA(generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::FA:
    FA(1.0 / sqrt((double)CE_Tnum), 0.5, 0.5, max_generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::CMAES:
    CMAES(generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::ADE:
    ADE(generation, max_generation, p_start, p_end);
    break;
  // 控制说明：处理这一算法或动作分支，完成后退出当前分派。
  case SearchAlg::BA:
    BA(p_start, p_end);
    break;
  default:
    DE(0.5, 1, 0.5, p_start, p_end);
    break;
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::RunAdaptiveGlobalSearch(int generation, int max_generation,
                                       int p_start, int p_end) {
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    IndividualSnapshot<double> snapshot = SnapshotIndividual(i);
    SearchAlg algorithm;
    algorithm = SelectAdaptiveGlobalAlgorithm();
    double old_fit = snapshot.pop_fit;
    double old_gbest = gbest_fit;

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    RunSelectedGlobalSearch(algorithm, generation, max_generation, i, i + 1,
                            true);
    RestorePopIndividual(i, snapshot);
    Evaluation(1, i, i + 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (algorithm == SearchAlg::ACO)
      phe_updating(i, i + 1);

    // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
    double local_gain = 0.0;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (newpop_fit[i] < old_fit)
      local_gain = (old_fit - newpop_fit[i]) / (fabs(old_fit) + 1e-12);
    double elite_gain = 0.0;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (newpop_fit[i] < old_gbest)
      elite_gain = (old_gbest - newpop_fit[i]) / (fabs(old_gbest) + 1e-12);
    double reward = local_gain + 2.0 * elite_gain;
    UpdateAdaptiveGlobalAlgorithm(algorithm, reward);
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::RunGlobalSearch(int configured_alg, int generation,
                               int max_generation, int p_start, int p_end) {
  SearchAlg algorithm = ConfiguredSearchAlg(configured_alg);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (algorithm == SearchAlg::Adaptive) {
    RunAdaptiveGlobalSearch(generation, max_generation, p_start, p_end);
  } else {
    RunSelectedGlobalSearch(algorithm, generation, max_generation, p_start,
                            p_end, false);
    int eval_end = p_end;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (algorithm == SearchAlg::ADE && Nvar > 500000 &&
        ADE_SKIP_EVAL_IMMIGRANTS > 0) {
      int skip_count = ADE_SKIP_EVAL_IMMIGRANTS;
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (skip_count > eval_end - p_start - 1)
        skip_count = max(0, eval_end - p_start - 1);
      eval_end -= skip_count;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int i = eval_end; i < p_end; i++)
        newpop_fit[i] = pop_fit[i];
    }
    Evaluation(1, p_start, eval_end);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (algorithm == SearchAlg::ACO)
      phe_updating(p_start, p_end);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (algorithm == SearchAlg::ADE)
      FinalizeADEGeneration(generation, max_generation, p_start, p_end);
  }
}

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
MultiMet::MultiMet(int psize, int nn, double lb, double ub, int c_num,
                   int e_num, int d_num, int ce_tnum, int m_jnum, int m_optnum,
                   FF evaluate)
    : Population(psize, nn, lb, ub), Cnum(c_num), Enum(e_num), Dnum(d_num),
      CE_Tnum(ce_tnum), M_Jnum(m_jnum), M_OPTnum(m_optnum) {
  EvaluFunc = evaluate;
  SearchTotal = 0;
  const bool lightweight_large_scale = Nvar >= 10000000;

  // Prob
#if VERBOSE_OUTPUT
  cout << "Allocating problem memory: Enum=" << Enum << ", Cnum=" << Cnum
       << ", Dnum=" << Dnum << ", CE_Tnum=" << CE_Tnum << ", Nvar=" << Nvar
       << endl;
#endif
  CETask_Property = new CETask[CE_Tnum];
  MTask_Time = new double[M_Jnum * M_OPTnum];
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (CE_Tnum >= 1000000) {
    EtoD_Distance = nullptr;
    DtoD_Distance = nullptr;
  } else {
    EtoD_Distance = CreateDistanceMatrix(Enum, Dnum);
    DtoD_Distance = CreateDistanceMatrix(Dnum, Dnum);
  }
  AvailDeviceList = new vector<int>[M_Jnum * M_OPTnum];
  EnergyList = new double[11];

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  CloudDevices = new vector<int>[Cnum];
  EdgeDevices = new vector<int>[Enum];
  CloudLoad = new vector<int>[Cnum];
  EdgeLoad = new vector<int>[Enum];
  DeviceLoad = new vector<int>[Dnum];
  CETask_coDevice = new vector<int>[CE_Tnum];
  Edge_Device_comm = new map<int, double>[Enum];
  ST = CreateMatrix(M_Jnum, M_OPTnum);
  ET = CreateMatrix(M_Jnum, M_OPTnum);
  CE_ST = new double[CE_Tnum];
  CE_ET = new double[CE_Tnum];

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if VERBOSE_OUTPUT
  cout << "Problem memory allocated" << endl;
#endif

  // PSO
  ibest = CreateMatrix(Popsize, Nvar);
  velocity = lightweight_large_scale ? nullptr : CreateMatrix(Popsize, Nvar);
  ibest_fit = new double[Popsize];
  AC1 = new double[Popsize];
  AC2 = new double[Popsize];
  AW = new double[Popsize];
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (Nvar <= kMaxOaDim) {
    OArow = Nvar + 1;
    OA = new int*[OArow];
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < OArow; i++)
      OA[i] = new int[Nvar];
  } else {
    OArow = 0;
    OA = nullptr;
  }

  // ACO
  ant_tao = lightweight_large_scale ? nullptr : CreateMatrix(Popsize, Nvar + 2);

  // ABCA
  trial = new int[Popsize];
  pr = new double[Popsize];

  // VNS
  neigh = new int[Popsize];

  // CMAES

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  weights = lightweight_large_scale ? nullptr : new double[Nvar];
  xstart = lightweight_large_scale ? nullptr : new double[Nvar];
  stddev = lightweight_large_scale ? nullptr : new double[Nvar];

  // PBILc
  PB_center = lightweight_large_scale ? nullptr : new double[Nvar];
  PB_sigma = lightweight_large_scale ? nullptr : new double[Nvar];

  // BATA
  BAT_r = new double[Popsize];
  BAT_A = new double[Popsize];
  BAT_v = lightweight_large_scale ? nullptr : CreateMatrix(Popsize, Nvar);

  // FA
  I = new double[Popsize];
  Index = new int[Popsize];

  // BA
  ngh = new double[Popsize];
  ngh_decay_count = new int[Popsize];

  // meme number
  Ind_meme = new int[Popsize];
  SubDecBase =
      lightweight_large_scale ? nullptr : CreateMatrix(Popsize, Nvar + 2);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  ade_active_size = Popsize;
  ade_policy = default_seri_policy();
  meme_policy = default_meme_policy();
  ade_path_center = lightweight_large_scale ? nullptr : new double[Nvar];
  ade_recent_path = lightweight_large_scale ? nullptr : new double[Nvar];
  ade_path_ready = false;
  ade_memory_pos = 0;
  ade_stagnant_generations = 0;
  ade_last_best = 1e300;
  ade_runtime_ready = false;
  meme_last_best = 1e300;
  meme_stagnant_generations = 0;
  ResetMemePolicyState();
  ResetSSALSState();
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!lightweight_large_scale)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Nvar; i++) {
      ade_path_center[i] = 0.0;
      ade_recent_path[i] = 0.0;
    }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
MultiMet::~MultiMet() {
  int i;
  // Prob
  for (i = 0; i < CE_Tnum; i++) {
    CETask_Property[i].Precedence.clear();
    CETask_Property[i].Interact.clear();
    CETask_Property[i].Start_Pre.clear();
    CETask_Property[i].End_Pre.clear();
    CETask_Property[i].AvailEdgeServerList.clear();
  }
  delete[] CETask_Property;
  delete[] MTask_Time;
  DeleteDistanceMatrix(EtoD_Distance, Enum);
  DeleteDistanceMatrix(DtoD_Distance, Dnum);
  delete[] AvailDeviceList;
  delete[] EnergyList;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  delete[] CloudDevices;
  delete[] EdgeDevices;
  delete[] CloudLoad;
  delete[] EdgeLoad;
  delete[] DeviceLoad;
  delete[] CETask_coDevice;
  delete[] Edge_Device_comm;
  DeleteMatrix(ST, M_Jnum);
  DeleteMatrix(ET, M_Jnum);
  delete[] CE_ST;
  delete[] CE_ET;

  // PSO
  DeleteMatrix(ibest, Popsize);
  DeleteMatrix(velocity, Popsize);
  delete[] ibest_fit;
  delete[] AC1;
  delete[] AC2;
  delete[] AW;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = 0; i < OArow; i++)
    delete[] OA[i];
  delete[] OA;

  // ACO
  DeleteMatrix(ant_tao, Popsize);

  // ABCA
  delete[] trial;
  delete[] pr;

  // VNS
  delete[] neigh;

  // CMAES
  delete[] xstart;
  delete[] stddev;
  delete[] weights;

  // PBILc
  delete[] PB_center;
  delete[] PB_sigma;

  // BATA
  delete[] BAT_r;
  delete[] BAT_A;
  DeleteMatrix(BAT_v, Popsize);

  // BA
  delete[] ngh;
  delete[] ngh_decay_count;

  // meme number
  delete[] Ind_meme;
  DeleteMatrix(SubDecBase, Popsize);
  delete[] ade_path_center;
  delete[] ade_recent_path;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::CreateOA() {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (OA == nullptr || OArow <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  int u = log((double)OArow) / log(2.0);
  int b;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < OArow; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < u; j++) {
      b = pow(2.0, j) - 1;
      int tmp = floor(i / pow(2.0, u - j - 1));
      OA[i][b] = tmp % 2;
    }
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < OArow; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < u; j++) {
      b = pow(2.0, j) - 1;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int s = 0; s < b; s++)
        OA[i][b + s + 1] = (OA[i][s] + OA[i][b]) % 2;
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::Initial() {
  // Prob
  fstream fs;
// 控制说明：选择当前编译配置对应的实现路径。
#if VERBOSE_OUTPUT
  cout << "Opening data file " << DATA_FILE_PATH << endl;
#endif
  fs.open(DATA_FILE_PATH, ios::in);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!fs) {
    cerr << "Failed to open data file: " << DATA_FILE_PATH << endl;
    exit(1);
  }
// 控制说明：选择当前编译配置对应的实现路径。
#if VERBOSE_OUTPUT
  cout << "Loading EtoD_Distance" << endl;
#endif
  double distance_value = 0.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (EtoD_Distance == nullptr) {
    string magic;
    ReadDataValue(fs, magic, "compact geometry header");
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (magic != "CED_COMPACT_V1")
      ReportDataReadFailure("compact geometry header");
    int edge_count = 0, device_count = 0;
    ReadDataValue(fs, edge_count, "compact edge count");
    ReadDataValue(fs, device_count, "compact device count");
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (edge_count != Enum || device_count != Dnum)
      ReportDataReadFailure("compact geometry dimensions");
    EdgeX.resize(Enum);
    EdgeY.resize(Enum);
    DeviceX.resize(Dnum);
    DeviceY.resize(Dnum);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Enum; ++i) {
      ReadDataValue(fs, EdgeX[i], "edge x", i);
      ReadDataValue(fs, EdgeY[i], "edge y", i);
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Dnum; ++i) {
      ReadDataValue(fs, DeviceX[i], "device x", i);
      ReadDataValue(fs, DeviceY[i], "device y", i);
    }
    CED_SetCompactGeometry(EdgeX.data(), EdgeY.data(), DeviceX.data(),
                           DeviceY.data(), Enum, Dnum);
  } else {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Enum; i++) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Dnum; j++) {
        ReadDataValue(fs, distance_value, "EtoD_Distance", i, j);
        EtoD_Distance[i][j] = EncodeDistance(distance_value);
      }
    }

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if VERBOSE_OUTPUT
    cout << "Loading DtoD_Distance" << endl;
#endif
    for (int i = 0; i < Dnum; i++) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Dnum; j++) {
        ReadDataValue(fs, distance_value, "DtoD_Distance", i, j);
        DtoD_Distance[i][j] = EncodeDistance(distance_value);
      }
    }
  }

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if VERBOSE_OUTPUT
  cout << "Loading task data" << endl;
#endif
  for (int i = 0; i < M_Jnum * M_OPTnum; i++)
    ReadDataValue(fs, MTask_Time[i], "MTask_Time", i);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  int vec_num = 0, value;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < CE_Tnum; i++) {
    ReadDataValue(fs, CETask_Property[i].Computation, "CETask.Computation", i);
    ReadDataValue(fs, CETask_Property[i].Communication, "CETask.Communication",
                  i);

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    ReadDataValue(fs, vec_num, "CETask.PrecedenceCount", i);
    CETask_Property[i].Precedence.clear();
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < vec_num; j++) {
      ReadDataValue(fs, value, "CETask.Precedence", i, j);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value >= 0 && value < CE_Tnum)
        CETask_Property[i].Precedence.push_back(value);
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    ReadDataValue(fs, vec_num, "CETask.InteractCount", i);
    CETask_Property[i].Interact.clear();
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < vec_num; j++) {
      ReadDataValue(fs, value, "CETask.Interact", i, j);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value >= 0 && value < CE_Tnum)
        CETask_Property[i].Interact.push_back(value);
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    ReadDataValue(fs, vec_num, "CETask.StartPreCount", i);
    CETask_Property[i].Start_Pre.clear();
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < vec_num; j++) {
      ReadDataValue(fs, value, "CETask.StartPre", i, j);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value >= 0 && value < CE_Tnum)
        CETask_Property[i].Start_Pre.push_back(value);
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    ReadDataValue(fs, vec_num, "CETask.EndPreCount", i);
    CETask_Property[i].End_Pre.clear();
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < vec_num; j++) {
      ReadDataValue(fs, value, "CETask.EndPre", i, j);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value >= 0 && value < CE_Tnum)
        CETask_Property[i].End_Pre.push_back(value);
    }

    // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
    ReadDataValue(fs, CETask_Property[i].Job_Constraints,
                  "CETask.JobConstraints", i);
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < M_Jnum; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < M_OPTnum; j++) {
      ReadDataValue(fs, vec_num, "AvailDeviceList.Count", i, j);
      AvailDeviceList[i * M_OPTnum + j].clear();
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int k = 0; k < vec_num; k++) {
        ReadDataValue(fs, value, "AvailDeviceList.Value", i, j);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (value >= 0 && value < Dnum)
          AvailDeviceList[i * M_OPTnum + j].push_back(value);
      }
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (AvailDeviceList[i * M_OPTnum + j].empty())
        AvailDeviceList[i * M_OPTnum + j].push_back(0);
    }
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < CE_Tnum; i++) {
    ReadDataValue(fs, vec_num, "CETask.AvailEdgeServerCount", i);
    CETask_Property[i].AvailEdgeServerList.clear();
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < vec_num; j++) {
      ReadDataValue(fs, value, "CETask.AvailEdgeServer", i, j);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (value >= 0 && value < Enum)
        CETask_Property[i].AvailEdgeServerList.push_back(value);
    }
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (CETask_Property[i].AvailEdgeServerList.empty())
      CETask_Property[i].AvailEdgeServerList.push_back(0);
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  fs.close();

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  fstream power_fs;
  power_fs.open(POWER_FILE_PATH, ios::in);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!power_fs) {
    cerr << "Failed to open power file: " << POWER_FILE_PATH << endl;
    exit(1);
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < 11; i++)
    ReadPowerValue(power_fs, EnergyList[i], "EnergyList", i);
  power_fs.close();
// 控制说明：选择当前编译配置对应的实现路径。
#if VERBOSE_OUTPUT
  cout << "Data file loaded" << endl;
#endif
  for (int i = 0; i < Popsize; i++)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; j++)
      newpop[i][j] = pop[i][j] = randval(Lbound, Ubound);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  if (velocity != nullptr)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Popsize; i++)
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Nvar; j++)
        velocity[i][j] = randval(Lbound, Ubound);

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if VERBOSE_OUTPUT
  cout << "Evaluating initial population" << endl;
#endif
  for (int i = 0; i < Popsize; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; j++)
      ibest[i][j] = pop[i][j];
    pop_fit[i] = EvaluFunc(
        pop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
        MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
        CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
        CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    newpop_fit[i] = pop_fit[i];

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    ibest_fit[i] = pop_fit[i];
  }
  worst_and_best();
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int j = 0; j < Nvar; j++)
    gbest[j] = pop[cur_best][j];
  gbest_fit = pop_fit[cur_best];
  CRfit();
  CRold = CRnew;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  if (Nvar >= 10000000) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Popsize; ++i)
      Ind_meme[i] = 0;
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  }

  // pso
  ac1 = ac2 = 2;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++) {
    AC1[i] = AC2[i] = 2;
    AW[i] = 0.85;
  }
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (OA != nullptr)
    CreateOA();

  // aco
  for (int i = 0; i < Popsize; i++)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; j++)
      ant_tao[i][j] = randval(Lbound, Ubound);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    ant_tao[i][Nvar] =
        EvaluFunc(ant_tao[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                  CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                  AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                  CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
                  Edge_Device_comm, ST, ET, CE_ST, CE_ET);
  heap_sort(ant_tao, Popsize, Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++) {
    ant_tao[i][Nvar + 1] = exp(-pow(i, 2.0) / (2 * pow(0.35 * Popsize, 2.0))) /
                           (0.35 * Popsize * sqrt(2 * kPi));
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Popsize; i++)
    trial[i] = 0;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int i = 0; i < Popsize; i++)
    neigh[i] = rand() % Nvar;

  // CMAES
  CMAisdone = false;

  // PBILc
  num_of_elite = min(2, Popsize / 2);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Nvar; i++)
    PB_center[i] = pop[0][i];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Nvar; i++)
    PB_sigma[i] = fabs(Ubound - Lbound) / 2;

  // BATA
  for (int i = 0; i < Popsize; i++) {
    BAT_r[i] = randval(0.0, 1.0);
    BAT_A[i] = 1.0;
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; j++)
      BAT_v[i][j] = 0; // randval(Lbound, Ubound);

  // FA
  for (int i = 0; i < Popsize; i++) {
    I[i] = pop_fit[i];
    Index[i] = i;
  }

  // BA
  ne = Popsize / 5 * 2;
  nre = 100;
  stlim = 10;
  ngh_decay = 0.8;
  ngh_origin = (Ubound - Lbound) * 0.1;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++) {
    ngh[i] = ngh_origin;
    ngh_decay_count[i] = 0;
  }

  // meme number
  for (int i = 0; i < Popsize; i++)
    Ind_meme[i] = rand() % 4;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Popsize; i++)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar + 2; j++)
      SubDecBase[i][j] = 0;

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  ade_active_size = Popsize;
  ade_policy = default_seri_policy();
  meme_policy = default_meme_policy();
  ade_path_ready = false;
  ade_memory_pos = 0;
  ade_stagnant_generations = 0;
  ade_last_best = gbest_fit;
  ade_runtime_ready = false;
  meme_last_best = gbest_fit;
  meme_stagnant_generations = 0;
  ade_seri_pool.clear();
  ade_success_flags.clear();
  ResetMemePolicyState();
  ResetSSALSState();
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Nvar; i++) {
    ade_path_center[i] = 0.0;
    ade_recent_path[i] = 0.0;
  }
}

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
void MultiMet::Evaluation(bool evaluate_new_population, int p_start,
                          int p_end) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!evaluate_new_population) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = p_start; i < p_end; i++)
      pop_fit[i] = EvaluFunc(
          pop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
          MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
          CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
          CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
  } else {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = p_start; i < p_end; i++)
      newpop_fit[i] =
          EvaluFunc(newpop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                    CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                    AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                    CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
                    Edge_Device_comm, ST, ET, CE_ST, CE_ET);
  }
}

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
void MultiMet::select(int p_start, int p_end) {
  double* rfitness = new double[Popsize];
  double* cfitness = new double[Popsize];
  double sum = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    sum += 1000.0 / pop_fit[i];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    rfitness[i] = (1000.0 / pop_fit[i]) / sum;
  cfitness[0] = rfitness[0];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 1; i < Popsize; i++)
    cfitness[i] = cfitness[i - 1] + rfitness[i];

  // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    double p = randval(0.0, 1.0);
    int selected = Popsize - 1;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Popsize; j++) {
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (p <= cfitness[j]) {
        selected = j;
        break;
      }
    }

    // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
    for (int s = 0; s < sub_dim; s++) {
      int k = particle_dim_index(Nvar, i, s, 67);
      newpop[i][k] = pop[selected][k];
    }
  }
  delete[] cfitness;
  delete[] rfitness;
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::crossover(double pc, int p_start, int p_end) {
  int mem;
  int pos = 0;
  double p;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (mem = p_start; mem < p_end; mem++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (p_end - p_start <= 1) {
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (Popsize <= 1)
        continue;
      // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
      do {
        pos = rand() % Popsize;
      } while (pos == mem);
    } else {
      // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
      do {
        pos = p_start + rand() % (p_end - p_start);
      } while (pos == mem);
    }
    p = randval(0, 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (p < pc) // 若概率小于pc，执行交换子xover操作
    {
      xover(mem, pos);
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::xover(int one, int two) {
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int s = 0; s < sub_dim; s++) {
    int i = particle_dim_index(Nvar, one + two, s, 71);
    double r = randval(0, 1);
    double value = newpop[one][i] * r + (1 - r) * newpop[two][i];
    newpop[one][i] = bounded_particle_value(value, Lbound, Ubound);
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::mutate(double pm, int p_start, int p_end) {
  double p;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    p = randval(0.0, 1.0);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (p < pm) {
      int r = rand() % Nvar;
      newpop[i][r] = randval(Lbound, Ubound);
    }
  }
}

// 段落说明：实现 `MultiMet::GA`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::GA(double pc, double pm, int p_start, int p_end) {
  select(p_start, p_end);
  crossover(pc, p_start, p_end);
  mutate(pm, p_start, p_end);
}

// 段落说明：实现 `MultiMet::NGA`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::NGA(double pc, double pm, double mdis, int p_start, int p_end) {
  pop_niche(mdis, p_start, p_end);
  select(p_start, p_end);
  crossover(pc, p_start, p_end);
  mutate(pm, p_start, p_end);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::LGA(double pc, double pm, double step, int nst, int p_start,
                   int p_end) {
  (void)step;
  select(p_start, p_end);
  crossover(pc, p_start, p_end);
  mutate(pm, p_start, p_end);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++)
    localsearch(newpop[i], nst);
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::VAGA(int p_start, int p_end) {
  double vari = pop_random_variance(0, Popsize);
  double p = exp(-vari);
  select(p_start, p_end);
  crossover(1 - p, p_start, p_end);
  mutate(p, p_start, p_end);
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::EAGA(int p_start, int p_end) {
  double entr = pop_random_entropy(max(2, Popsize / 2), 0, Popsize);
  double p = exp(-entr);
  select(p_start, p_end);
  crossover(1 - p, p_start, p_end);
  mutate(p, p_start, p_end);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::PSO(double w, double c1, double c2, double max_ve, int p_start,
                   int p_end) {
  int i, j;
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      j = particle_dim_index(Nvar, i, s, 1);
      double r1 = randval(0, 1);
      double r2 = randval(0, 1);
      velocity[i][j] = w * velocity[i][j] +
                       c1 * r1 * (ibest[i][j] - newpop[i][j]) +
                       c2 * r2 * (gbest[j] - newpop[i][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (velocity[i][j] > max_ve)
        velocity[i][j] = max_ve;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (velocity[i][j] < -max_ve)
        velocity[i][j] = -max_ve;
      newpop[i][j] =
          bounded_particle_value(newpop[i][j] + velocity[i][j], Lbound, Ubound);
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::CPSO(double w, double c1, double c2, double max_ve, int p_start,
                    int p_end) {
  int i, j;
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      j = particle_dim_index(Nvar, i, s, 2);
      double r1 = randval(0, 1);
      double r2 = randval(0, 1);
      velocity[i][j] =
          0.729 * (w * velocity[i][j] + c1 * r1 * (ibest[i][j] - newpop[i][j]) +
                   c2 * r2 * (gbest[j] - newpop[i][j]));
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (velocity[i][j] > max_ve)
        velocity[i][j] = max_ve;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (velocity[i][j] < -max_ve)
        velocity[i][j] = -max_ve;
      newpop[i][j] =
          bounded_particle_value(newpop[i][j] + velocity[i][j], Lbound, Ubound);
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::Subgradient(double* theta, int q, double c_step,
                           double* subgrad) {
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Nvar; i++)
    subgrad[i] = 0.0;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const int sample_dim = min(Nvar, min(max(1, q), 16));
  const double step = max(1e-5, fabs(c_step));
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int s = 0; s < sample_dim; s++) {
    int dim = particle_dim_index(Nvar, 0, s, 8);
    double old_value = theta[dim];
    theta[dim] = bounded_particle_value(old_value + step, Lbound, Ubound);
    double plus = EvaluFunc(
        theta, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
        MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
        CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
        CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    theta[dim] = bounded_particle_value(old_value - step, Lbound, Ubound);
    double minus = EvaluFunc(
        theta, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
        MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
        CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
        CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    theta[dim] = old_value;
    subgrad[dim] = (plus - minus) / (2.0 * step + 1e-12);
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::SPSO(double w, double c1, double c2, double max_ve, int Gen,
                    int MaxGen, int p_start, int p_end) {
  (void)Gen;
  (void)MaxGen;
  int i, j;
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = p_start; i < p_end; i++) {
    double velnorm = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      j = particle_dim_index(Nvar, i, s, 3);
      double r1 = randval(0, 1);
      double r2 = randval(0, 1);
      velocity[i][j] = w * velocity[i][j] +
                       c1 * r1 * (ibest[i][j] - newpop[i][j]) +
                       c2 * r2 * (gbest[j] - newpop[i][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (velocity[i][j] > max_ve)
        velocity[i][j] = max_ve;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (velocity[i][j] < -max_ve)
        velocity[i][j] = -max_ve;
      velnorm += velocity[i][j] * velocity[i][j];
    }
    velnorm = sqrt(velnorm / max(1, sub_dim));
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      j = particle_dim_index(Nvar, i, s, 4);
      double pull = 0.45 * (gbest[j] - newpop[i][j]) + 0.55 * velocity[i][j];
      newpop[i][j] = bounded_particle_value(
          newpop[i][j] + pull / (1.0 + velnorm), Lbound, Ubound);
    }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::Cauchy_mutation(double* pp, int Gen, int MaxGen) {
  int bit = rand() % Nvar;
  double p = randval(0, 1);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (p < 0.5)
    pp[bit] += (Ubound - pp[bit]) *
               (1 - pow(randval(0, 1), (1 - (double)Gen / MaxGen)));
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    pp[bit] += (pp[bit] - Lbound) *
               (1 - pow(randval(0, 1), (1 - (double)Gen / MaxGen)));
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (pp[bit] > Ubound)
    pp[bit] = Ubound;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (pp[bit] < Lbound)
    pp[bit] = Lbound;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::CMPSO(double w, double c1, double c2, double max_ve, int Gen,
                     int MaxGen, int p_start, int p_end) {
  int i, j;
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      j = particle_dim_index(Nvar, i, s, 5);
      double r1 = randval(0, 1);
      double r2 = randval(0, 1);
      velocity[i][j] = w * velocity[i][j] +
                       c1 * r1 * (ibest[i][j] - newpop[i][j]) +
                       c2 * r2 * (gbest[j] - newpop[i][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (velocity[i][j] > max_ve)
        velocity[i][j] = max_ve;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (velocity[i][j] < -max_ve)
        velocity[i][j] = -max_ve;
      newpop[i][j] =
          bounded_particle_value(newpop[i][j] + velocity[i][j], Lbound, Ubound);
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (randval(0, 1) < 0.01)
      Cauchy_mutation(newpop[i], Gen, MaxGen);
  }
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
void MultiMet::APSO_1(double c1, double c2, double max_ve, int Gen, int MaxGen,
                      int p_start, int p_end) {
  double w = 0.9 - 0.5 * Gen / MaxGen;
  PSO(w, c1, c2, max_ve, p_start, p_end);
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
void MultiMet::APSO_2(double c1, double c2, double max_ve, int p_start,
                      int p_end) {
  double w = 0.5 + randval(0, 1) / 2;
  PSO(w, c1, c2, max_ve, p_start, p_end);
}

// 段落说明：实现 `MultiMet::APSO_3`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::APSO_3(double max_ve, int p_start, int p_end) {
  const int sample_dim = (Nvar < 64) ? Nvar : 64;
  double dg = 0.0, dmin = 1e100, dmax = 0.0;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Popsize; i++) {
    double tmp = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sample_dim; s++) {
      int k = (s * 9973 + i * 37) % Nvar;
      double diff = newpop[i][k] - gbest[k];
      tmp += diff * diff;
    }
    dg += sqrt(tmp * ((double)Nvar / sample_dim));
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Popsize; i++) {
    double di = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Popsize; j++) {
      double tmp = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int s = 0; s < sample_dim; s++) {
        int k = (s * 9973 + i * 37 + j * 101) % Nvar;
        double diff = newpop[i][k] - newpop[j][k];
        tmp += diff * diff;
      }
      di += sqrt(tmp * ((double)Nvar / sample_dim));
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (di > dmax)
      dmax = di;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (di < dmin)
      dmin = di;
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  double denom = dmax - dmin;
  double f = 0.5;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (fabs(denom) > 1e-12)
    f = ((dg / (Popsize > 1 ? Popsize - 1.0 : 1.0)) - dmin) / denom;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (f < 0.0)
    f = 0.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (f > 1.0)
    f = 1.0;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (f < 0.3) {
    ac1 += 0.08;
    ac2 -= 0.04;
  } else if (f > 0.75) {
    ac1 -= 0.06;
    ac2 += 0.08;
  } else {
    ac1 += 0.03;
    ac2 += 0.03;
  }
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ac1 < 0.5)
    ac1 = 0.5;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (ac1 > 2.8)
    ac1 = 2.8;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ac2 < 0.5)
    ac2 = 0.5;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (ac2 > 2.8)
    ac2 = 2.8;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ac1 + ac2 > 4.0) {
    double sum = ac1 + ac2;
    ac1 = 4.0 * ac1 / sum;
    ac2 = 4.0 * ac2 / sum;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double w = 0.45 + 0.45 * f;
  const int update_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < update_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 6);
      double r1 = randval(0, 1);
      double r2 = randval(0, 1);
      velocity[i][j] = w * velocity[i][j] +
                       ac1 * r1 * (ibest[i][j] - newpop[i][j]) +
                       ac2 * r2 * (gbest[j] - newpop[i][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (velocity[i][j] > max_ve)
        velocity[i][j] = max_ve;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (velocity[i][j] < -max_ve)
        velocity[i][j] = -max_ve;
      newpop[i][j] =
          bounded_particle_value(newpop[i][j] + velocity[i][j], Lbound, Ubound);
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::APSO_4(double max_ve, int Gen, int MaxGen, int p_start,
                      int p_end) {
  (void)MaxGen;
  const int sample_dim = (Nvar < 64) ? Nvar : 64;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    double gw = 0.0, gc1 = 0.0, gc2 = 0.0, fdist = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sample_dim; s++) {
      int j = (s * 9973 + i * 97 + Gen * 193) % Nvar;
      double diff = newpop[i][j] - gbest[j];
      double r1 = randval(0, 1);
      double r2 = randval(0, 1);
      fdist += diff * diff;
      gw += diff * velocity[i][j];
      gc1 += diff * r1 * (ibest[i][j] - newpop[i][j]);
      gc2 += diff * r2 * (gbest[j] - newpop[i][j]);
    }
    fdist = sqrt(fdist * ((double)Nvar / sample_dim));
    gw *= 2.0;
    gc1 *= 2.0;
    gc2 *= 2.0;

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    double alp = fdist / (gw * gw + gc1 * gc1 + gc2 * gc2 + 1.0);
    AW[i] -= alp * gw;
    AC1[i] -= alp * gc1;
    AC2[i] -= alp * gc2;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (AW[i] < 0.4)
      AW[i] = 0.4;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (AW[i] > 0.9)
      AW[i] = 0.9;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (AC1[i] < 0.5)
      AC1[i] = 0.5;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (AC1[i] > 2.5)
      AC1[i] = 2.5;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (AC2[i] < 0.5)
      AC2[i] = 0.5;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (AC2[i] > 2.5)
      AC2[i] = 2.5;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    double r1 = randval(0, 1);
    double r2 = randval(0, 1);
    const int update_dim = particle_subspace_dim(Nvar);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < update_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 7);
      velocity[i][j] = AW[i] * velocity[i][j] +
                       AC1[i] * r1 * (ibest[i][j] - newpop[i][j]) +
                       AC2[i] * r2 * (gbest[j] - newpop[i][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (velocity[i][j] > max_ve)
        velocity[i][j] = max_ve;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (velocity[i][j] < -max_ve)
        velocity[i][j] = -max_ve;
      newpop[i][j] =
          bounded_particle_value(newpop[i][j] + velocity[i][j], Lbound, Ubound);
    }
  }
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
void MultiMet::APSO_5(double max_ve, int Gen, int MaxGen, int p_start,
                      int p_end) {
  APSO_4(max_ve, Gen, MaxGen, p_start, p_end);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::Orthogonal_P(double* P0, double w, double c, int ppn) {
  (void)w;
  (void)c;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int s = 0; s < particle_subspace_dim(Nvar); s++) {
    int j = particle_dim_index(Nvar, ppn, s, 9);
    P0[j] = (randval(0, 1) < 0.5) ? ibest[ppn][j] : gbest[j];
  }
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
void MultiMet::OLPSO(double w, double c, double max_ve, int p_start,
                     int p_end) {
  PSO(w, c, c, max_ve, p_start, p_end);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::newpop_worst_best(int& w, int& b, int p_start, int p_end) {
  w = p_start;
  b = p_start;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (p_start >= p_end)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start + 1; i < p_end; i++) {
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (newpop_fit[i] > newpop_fit[w])
      w = i;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (newpop_fit[i] < newpop_fit[b])
      b = i;
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::HS(double srate, double trate, double bw, int p_start,
                  int p_end) {
  int ww = 0, bb = 0;
  const int sub_dim = particle_subspace_dim(Nvar);
  vector<int> changed_dim(sub_dim);
  vector<double> old_value(sub_dim);
  newpop_worst_best(ww, bb, p_start, p_end);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 10);
      changed_dim[s] = j;
      old_value[s] = newpop[ww][j];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (randval(0, 1) < srate)
        newpop[ww][j] = newpop[rand() % Popsize][j];
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        newpop[ww][j] = randval(Lbound, Ubound);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (randval(0, 1) < trate)
        newpop[ww][j] += randval(0, 1) * bw;
      newpop[ww][j] = clamp_particle_value(newpop[ww][j], Lbound, Ubound);
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    double trial_fit =
        EvaluFunc(newpop[ww], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                  CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                  AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                  CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
                  Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (trial_fit < newpop_fit[ww]) {
      newpop_fit[ww] = trial_fit;
      newpop_worst_best(ww, bb, p_start, p_end);
    } else {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int s = 0; s < sub_dim; s++)
        newpop[ww][changed_dim[s]] = old_value[s];
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::path_finding(double epsl, int p_start, int p_end) {
  int l = 0;
  double psum = 0.0;
  double* rp = new double[Popsize];
  double* cp = new double[Popsize];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    psum += ant_tao[i][Nvar + 1];
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (psum <= 1e-300)
    psum = 1.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    rp[i] = ant_tao[i][Nvar + 1] / psum;
  cp[0] = rp[0];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 1; i < Popsize; i++)
    cp[i] = cp[i - 1] + rp[i];

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const int sub_dim = (Nvar < 24) ? Nvar : 24;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    double p = randval(0.0, 1.0);
    l = Popsize - 1;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Popsize; j++) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (p <= cp[j]) {
        l = j;
        break;
      }
    }

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    for (int d = 0; d < sub_dim; d++) {
      int j = rand() % Nvar;
      double spread = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int k = 0; k < Popsize; k++)
        spread += fabs(ant_tao[k][j] - ant_tao[l][j]);
      spread = epsl * spread / (Popsize > 1 ? Popsize - 1.0 : 1.0);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (spread < 1e-4)
        spread = 1e-4;

      // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
      double direction = 0.65 * (ant_tao[l][j] - newpop[i][j]) +
                         0.35 * (gbest[j] - newpop[i][j]);
      newpop[i][j] += direction + randnorm(0.0, spread);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (newpop[i][j] < Lbound)
        newpop[i][j] = Lbound;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (newpop[i][j] > Ubound)
        newpop[i][j] = Ubound;
    }
  }
  delete[] rp;
  delete[] cp;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::phe_updating(int p_start, int p_end) {
  double maxTao = -1e300;
  int maxIndex = 0;
  bool sameflag = false;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    sameflag = false;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Popsize; j++)
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (fabs(ant_tao[j][Nvar] - newpop_fit[i]) <=
          1e-12 * (fabs(newpop_fit[i]) + 1.0))
        sameflag = true;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (sameflag == false) {
      maxTao = -1e300;
      maxIndex = 0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Popsize; j++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (ant_tao[j][Nvar] > maxTao) {
          maxTao = ant_tao[j][Nvar];
          maxIndex = j;
        }
      }
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (newpop_fit[i] < maxTao) {
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int j = 0; j < Nvar; j++)
          ant_tao[maxIndex][j] = newpop[i][j];
        ant_tao[maxIndex][Nvar] = newpop_fit[i];
      }
    }
  }
  heap_sort(ant_tao, Popsize, Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++) {
    ant_tao[i][Nvar + 1] = exp(-pow(i, 2.0) / (2 * pow(0.35 * Popsize, 2.0))) /
                           (0.35 * Popsize * sqrt(2 * kPi));
  }
}

// 段落说明：实现 `MultiMet::ACO`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::ACO(double epsl, int p_start, int p_end) {
  path_finding(epsl, p_start, p_end);
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::chaos(double** incpop, int chaos_n) {
  int i, j;
  int l, point;
  double* x1 = new double[chaos_n + 1];
  double* x2 = new double[chaos_n + 1];
  x1[0] = randval(0, 1);
  x2[0] = randval(0, 1);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = 1; i < chaos_n; i++) {
    x1[i] = 4 * x1[i - 1] * (1 - x1[i - 1]);
    x2[i] = 4 * x2[i - 1] * (1 - x2[i - 1]);
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = 0; i < chaos_n; i++) {
    l = rand() % Popsize;
    point = (int)(x1[i] * Nvar) % Nvar;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (j = 0; j < Nvar; j++)
      incpop[l][j] = gbest[j];
    incpop[l][point] = Lbound + x2[i] * (Ubound - Lbound);
  }
  delete[] x1;
  delete[] x2;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::COA(int chaos_n, int p_start, int p_end) {
  vector<int> changed_dim(max(1, chaos_n));
  vector<double> old_value(max(1, chaos_n));
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    double x1 = randval(0, 1);
    double x2 = randval(0, 1);
    int changed = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < chaos_n; j++) {
      x1 = 4 * x1 * (1 - x1);
      x2 = 4 * x2 * (1 - x2);
      int point = (int)(x1 * Nvar) % Nvar;
      changed_dim[changed] = point;
      old_value[changed] = newpop[i][point];
      newpop[i][point] = Lbound + x2 * (Ubound - Lbound);
      changed++;
    }
    double ff = EvaluFunc(
        newpop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
        MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
        CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
        CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (ff < newpop_fit[i]) {
      newpop_fit[i] = ff;
    } else {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < changed; j++)
        newpop[i][changed_dim[j]] = old_value[j];
    }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::differential_mutate(double F, int S, int p_start, int p_end) {
  int x[5];
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    x[0] = rand() % Popsize;
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    do {
      x[1] = rand() % Popsize;
    } while (x[1] == x[0]);
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    do {
      x[2] = rand() % Popsize;
    } while (x[2] == x[1] || x[2] == x[0]);
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    do {
      x[3] = rand() % Popsize;
    } while (x[3] == x[2] || x[3] == x[1] || x[3] == x[0]);
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    do {
      x[4] = rand() % Popsize;
    } while (x[4] == x[3] || x[4] == x[2] || x[4] == x[1] || x[4] == x[0]);

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 11 + S);
      double value;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (S == 1)
        value = gbest[j] + F * (ibest[x[0]][j] - ibest[x[1]][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (S == 2)
        value = ibest[x[0]][j] + F * (ibest[x[1]][j] - ibest[x[2]][j]) +
                F * (ibest[x[3]][j] - ibest[x[4]][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (S == 3)
        value = gbest[j] + F * (ibest[x[0]][j] - ibest[x[1]][j]) +
                F * (ibest[x[2]][j] - ibest[x[3]][j]);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (S == 4)
        value = ibest[i][j] + F * (gbest[j] - ibest[i][j]) +
                F * (ibest[x[0]][j] - ibest[x[1]][j]);
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        value = ibest[x[0]][j] + F * (ibest[x[1]][j] - ibest[x[2]][j]);
      newpop[i][j] = clamp_particle_value(value, Lbound, Ubound);
    }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::differential_crossover(double cr, int p_start, int p_end) {
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    int forced = particle_dim_index(Nvar, i, rand() % sub_dim, 17);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 17);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (randval(0, 1) > cr && j != forced)
        newpop[i][j] = pop[i][j];
    }
  }
}

// 段落说明：实现 `MultiMet::DE`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::DE(double F, int S, double cr, int p_start, int p_end) {
  differential_mutate(F, S, p_start, p_end);
  differential_crossover(cr, p_start, p_end);
}

// 段落说明：实现 `MultiMet::levy_cuckoo`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::levy_cuckoo(int p_start, int p_end) {
  const int sub_dim = particle_subspace_dim(Nvar);
  const double beta = 1.5;
  double a = randval(0, 1), b = randval(1e-5, 1);
  double levy_sigma =
      pow((a * sin(kPi * beta / 2) / (b * beta * pow(2.0, (beta - 1) / 2.0))),
          (1.0 / beta));

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 23);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (randval(0, 1) >= 0.35)
        continue;

      // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
      double u = randval(0, 1) * levy_sigma;
      double v = randval(1e-5, 1);
      double step = u / (pow(v, 1.0 / beta) + 1e-5);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (step > 1.0)
        step = 1.0;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (step < -1.0)
        step = -1.0;

      // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
      double stepsize = 0.1 * step * (ibest[i][j] - pop[i][j]);
      newpop[i][j] = bounded_particle_value(
          pop[i][j] + stepsize * randval(0, 1), Lbound, Ubound);
    }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::nest_discover(double pa, int p_start, int p_end) {
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    int r1 = rand() % Popsize;
    int r2 = rand() % Popsize;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (r1 == r2)
      continue;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    for (int s = 0; s < sub_dim; s++) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (randval(0, 1) >= pa)
        continue;
      int j = particle_dim_index(Nvar, i, s, 29);
      double dis = newpop[r1][j] - newpop[r2][j];
      newpop[i][j] = bounded_particle_value(newpop[i][j] + randval(0, 1) * dis,
                                            Lbound, Ubound);
    }
  }
}

// 段落说明：实现 `MultiMet::CSA`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::CSA(double pa, int p_start, int p_end) {
  levy_cuckoo(p_start, p_end);
  nest_discover(pa, p_start, p_end);
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::EmployedBee(int pn, double* tmp, double** pp) {
  (void)tmp;
  int neighbor;
  // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
  do {
    neighbor = rand() % Popsize;
  } while (neighbor == pn);

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  int para2change = rand() % Nvar;
  double base = pp[pn][para2change];
  double value = base + (pp[neighbor][para2change] - base) * randval(-1, 1);
  newpop[pn][para2change] = bounded_particle_value(value, Lbound, Ubound);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::OnlookerBee(int p_start, int p_end) {
  double maxf = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    pr[i] = exp(newpop_fit[i] / 1000);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (pr[i] > 1e10)
      pr[i] = 1e10;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (pr[i] > maxf)
      maxf = pr[i];
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++)
    pr[i] = (maxf != 0) ? 0.9 * pr[i] / maxf + 0.1 : 1.0;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::ScoutBee(int limit, int p_start, int p_end) {
  int maxindex = p_start;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start + 1; i < p_end; i++)
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (trial[i] > trial[maxindex])
      maxindex = i;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (trial[maxindex] <= limit)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int s = 0; s < sub_dim; s++) {
    int j = particle_dim_index(Nvar, maxindex, s, 31);
    newpop[maxindex][j] = randval(Lbound, Ubound);
  }
  trial[maxindex] = 0;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::ABCA(int limit, int p_start, int p_end) {
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    EmployedBee(i, nullptr, pop);
    trial[i]++;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  OnlookerBee(p_start, p_end);
  int attempts = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end && attempts < (p_end - p_start);
       i++, attempts++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (randval(0, 1) < pr[i])
      EmployedBee(i, nullptr, newpop);
  }

  // 段落说明：输出可审计的运行信息、指标或错误原因。
  ScoutBee(limit, p_start, p_end);
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::plant_growth(int p_start, int p_end) // dis < Ubound - Lbound
{
  const int sub_dim = (Nvar < 24) ? Nvar : 24;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    double r = randval(0.0, 1.0);
    double elite_weight =
        (r < 0.45) ? randval(0.20, 0.65) : randval(0.05, 0.35);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < sub_dim; d++) {
      int j = rand() % Nvar;
      double target;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (r < 0.45)
        target = gbest[j];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (r < 0.70)
        target = pop[cur_best][j];
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (r < 0.90)
        target = ibest[i][j];
      // 控制说明：条件不成立时执行互斥的备用处理路径。
      else
        target = randval(Lbound, Ubound);

      // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
      double jitter = (r < 0.90) ? randval(-0.03, 0.03) : 0.0;
      newpop[i][j] += elite_weight * (target - newpop[i][j]) + jitter;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (newpop[i][j] > Ubound)
        newpop[i][j] = Ubound;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (newpop[i][j] < Lbound)
        newpop[i][j] = Lbound;
    }
  }
}

// 段落说明：实现 `MultiMet::POA`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::POA(int p_start, int p_end) {
  plant_growth(p_start, p_end);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::localsearch(double* pp, int nst) // same with bit climbing
{
  int steps = nst;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (steps < 1)
    steps = 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (steps > 8)
    steps = 8;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  double best_fit = EvaluFunc(
      pp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
      MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
      CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
      CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int s = 0; s < steps; s++) {
    int bit = rand() % Nvar;
    double old_value = pp[bit];
    pp[bit] = randval(Lbound, Ubound);
    double trial_fit = EvaluFunc(
        pp, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
        MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
        CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
        CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (trial_fit < best_fit)
      best_fit = trial_fit;
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      pp[bit] = old_value;
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::ILS(int nst, int p_start, int p_end) {
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 37);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (randval(0, 1) >= 0.5)
        continue;
      int mate = rand() % Popsize;
      newpop[i][j] = bounded_particle_value(
          pop[i][j] + randval(0, 1) * (pop[mate][j] - pop[i][j]), Lbound,
          Ubound);
    }
    localsearch(newpop[i], nst);
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::VNS(int nst, int p_start, int p_end) {
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    int bit = neigh[i] % Nvar;
    int mate = rand() % Popsize;
    newpop[i][bit] = bounded_particle_value(
        pop[i][bit] + randval(0, 1) * (pop[mate][bit] - pop[i][bit]), Lbound,
        Ubound);
    localsearch(newpop[i], nst);
    neigh[i] = (neigh[i] + 1) % Nvar;
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::GRASP(double alfa, int nst, int p_start, int p_end) {
  double threshold =
      pop_fit[cur_best] + alfa * (pop_fit[cur_worst] - pop_fit[cur_best]);
  int elite_count = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (pop_fit[i] <= threshold)
      elite_count++;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (elite_count <= 0)
    elite_count = 1;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    int pick = rand() % elite_count;
    int source = cur_best;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0, seen = 0; k < Popsize; k++) {
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (pop_fit[k] <= threshold) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (seen == pick) {
          source = k;
          break;
        }
        seen++;
      }
    }

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 41);
      newpop[i][j] = pop[source][j];
    }
    localsearch(newpop[i], nst);
  }
}

// 段落说明：排序或定位最优/最差元素，为选择、统计或更新提供确定顺序。
void MultiMet::PBILC(int p_start, int p_end, double learn_rate) {
  pop_heap_sort(Popsize);
  const int sub_dim = particle_subspace_dim(Nvar);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int s = 0; s < sub_dim; s++) {
    int j = particle_dim_index(Nvar, p_start + p_end, s, 43);
    double old_center = PB_center[j];
    double old_sigma = PB_sigma[j];

    // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
    double center = pop[0][j] + pop[1][j] - pop[Popsize - 1][j];
    PB_center[j] = (1 - learn_rate) * old_center + learn_rate * center;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    double average = 0.0, sum = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k < num_of_elite; k++)
      average += pop[k][j];
    average /= num_of_elite;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k < num_of_elite; k++)
      sum += pow(pop[k][j] - average, 2);
    sum /= num_of_elite;

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    PB_sigma[j] = (1 - learn_rate) * old_sigma + learn_rate * sqrt(sum);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (PB_sigma[j] < 1e-6)
      PB_sigma[j] = 1e-6;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 47);
      double value = PB_center[j] + PB_sigma[j] * gauss();
      newpop[i][j] = bounded_particle_value(value, Lbound, Ubound);
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::BATA(int gen, int p_start, int p_end) {
  pop_heap_sort(Popsize);
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    bool pulse = randval(0.0, 1.0) > BAT_r[i];
    int elite_limit = Popsize / 2;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (elite_limit < 1)
      elite_limit = 1;

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    for (int s = 0; s < sub_dim; s++) {
      int j = particle_dim_index(Nvar, i, s, 53);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (pulse) {
        int elite = rand() % elite_limit;
        newpop[i][j] = pop[elite][j] + randval(-1.0, 1.0) * BAT_A[i];
      } else {
        BAT_v[i][j] += (pop[i][j] - pop[cur_best][j]) * randval(0.0, 1.0);
        newpop[i][j] = pop[i][j] + BAT_v[i][j];
      }
      newpop[i][j] = bounded_particle_value(newpop[i][j], Lbound, Ubound);
    }

    // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
    BAT_A[i] *= 0.995;
    BAT_r[i] *= (1 - exp(-0.95 * (gen + 1)));
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::FA(double gama, double alpha0, double betamin, int MGEN,
                  int p_start, int p_end) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (alpha == alpha0)
    alpha = pow((pow(10.0, -4.0) / 0.9), 1.0 / MGEN) * alpha0;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    alpha = pow((pow(10.0, -4.0) / 0.9), 1.0 / MGEN) * alpha;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (alpha < 1e-5)
    alpha = 0.5;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const int sub_dim = particle_subspace_dim(Nvar);
  double scale = abs(Ubound - Lbound);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Popsize; j++) {
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (i == j || pop_fit[i] <= pop_fit[j])
        continue;

      // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
      double r = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int s = 0; s < sub_dim; s++) {
        int k = particle_dim_index(Nvar, i + j, s, 59);
        double diff = newpop[i][k] - pop[j][k];
        r += diff * diff;
      }
      r = sqrt(r / sub_dim);
      double beta = (1.0 - betamin) * exp(-gama * r * r) + betamin;

      // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
      for (int s = 0; s < sub_dim; s++) {
        int k = particle_dim_index(Nvar, i + j, s, 61);
        double noise = alpha * (randval(0.0, 2.0) - 1.0) * scale;
        double value = newpop[i][k] * (1.0 - beta) + pop[j][k] * beta + noise;
        newpop[i][k] = bounded_particle_value(value, Lbound, Ubound);
      }
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::pop_heap_adjust(int s, int len) {
  double* temp = pop[s];
  double temp_fit = pop_fit[s];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 2 * s + 1; i < len; i = 2 * i + 1) {
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (i < (len - 1) && pop_fit[i] < pop_fit[i + 1])
      i++;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (temp_fit > pop_fit[i])
      break;
    pop[s] = pop[i];
    pop_fit[s] = pop_fit[i];
    s = i;
  }
  pop[s] = temp;
  pop_fit[s] = temp_fit;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::pop_heap_sort(int len) {
  int i;
  double* temp;
  double temp_fit;
  double p_ngh;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = len / 2 - 1; i >= 0; i--)
    pop_heap_adjust(i, len);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = len - 1; i > 0; i--) {
    temp = pop[0];
    temp_fit = pop_fit[0];
    p_ngh = ngh[0];

    // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
    pop[0] = pop[i];
    pop_fit[0] = pop_fit[i];
    ngh[0] = ngh[i];

    // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
    ngh[i] = p_ngh;
    pop[i] = temp;
    pop_fit[i] = temp_fit;
    pop_heap_adjust(0, i);
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::newpop_heap_adjust(int s, int len) {
  double* temp = newpop[s];
  double temp_fit = newpop_fit[s];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 2 * s + 1; i < len; i = 2 * i + 1) {
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (i < (len - 1) && newpop_fit[i] < newpop_fit[i + 1])
      i++;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (temp_fit > newpop_fit[i])
      break;
    newpop[s] = newpop[i];
    newpop_fit[s] = newpop_fit[i];
    s = i;
  }
  newpop[s] = temp;
  newpop_fit[s] = temp_fit;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::newpop_heap_sort(int len) {
  int i;
  double* temp;
  double temp_fit;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = len / 2 - 1; i >= 0; i--)
    newpop_heap_adjust(i, len);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = len - 1; i > 0; i--) {
    temp = newpop[0];
    temp_fit = newpop_fit[0];
    newpop[0] = newpop[i];
    newpop_fit[0] = newpop_fit[i];
    newpop[i] = temp;
    newpop_fit[i] = temp_fit;
    newpop_heap_adjust(0, i);
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::pop_niche(double mdis, int p_start, int p_end) {
  const int sub_dim = particle_subspace_dim(Nvar);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = i + 1; j < p_end; j++) {
      double dis = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int s = 0; s < sub_dim; s++) {
        int k = particle_dim_index(Nvar, i + j, s, 73);
        double diff = pop[i][k] - pop[j][k];
        dis += diff * diff;
      }
      dis = sqrt(dis / sub_dim);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (dis < mdis) {
        // 控制说明：依据目标值决定接受、最优更新或审计路径。
        if (pop_fit[i] > pop_fit[j])
          pop_fit[i] = abs(pop_fit[i] * 100);
        // 控制说明：条件不成立时执行互斥的备用处理路径。
        else
          pop_fit[j] = abs(pop_fit[j] * 100);
      }
    }
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::newpop_localstep(double step, int nst, int p_start, int p_end) {
  double Fave = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++)
    Fave += newpop_fit[i];
  Fave /= Popsize;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  int steps = nst;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (steps < 1)
    steps = 1;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (steps > 8)
    steps = 8;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = p_start; i < p_end; i++) {
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (randval(0, 1) >= 1 / (1 + exp(newpop_fit[i] - Fave)))
      continue;

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    int bit = rand() % Nvar;
    double base = newpop[i][bit];
    double best_value = base;
    double best_fit = newpop_fit[i];
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < steps; j++) {
      double candidate = base - j * step;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (candidate < Lbound)
        candidate = randval(Lbound, Ubound);
      newpop[i][bit] = candidate;
      double temp_fit =
          EvaluFunc(newpop[i], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                    CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                    AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                    CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
                    Edge_Device_comm, ST, ET, CE_ST, CE_ET);
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (temp_fit < best_fit) {
        best_fit = temp_fit;
        best_value = candidate;
        break;
      }
    }
    newpop[i][bit] = best_value;
    newpop_fit[i] = best_fit;
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
void MultiMet::CMAES(int firstrun, int p_start, int p_end) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!CMAisdone) {
    CMAES_parametersetting();
    CMAES_initial();
    CMAisdone = true;
  }
  CMAES_updateDistribution(firstrun);
  CMAES_sampleGenerate(firstrun, p_start, p_end);
}

// 段落说明：实现 `MultiMet::CMAES_parametersetting`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::CMAES_parametersetting() {

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Nvar; i++)
    xstart[i] = 0.5 * (Lbound + Ubound);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Nvar; i++)
    stddev[i] = 0.3 * (Ubound - Lbound);
  lambda = min(Popsize, 4 + (int)floor(3 * log(Nvar)));
  mu = lambda / 2;
  // setweight
  for (int i = 0; i < mu; ++i)
    weights[i] = log(mu + 1.) - log(i + 1.);
  double s1 = 0., s2 = 0.;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < mu; ++i) {
    s1 += weights[i];
    s2 += weights[i] * weights[i];
  }
  mueff = s1 * s1 / s2;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < mu; ++i)
    weights[i] /= s1;
  cs = (mueff + 2.) / (Nvar + mueff + 3.);
  ccumcov = (4. + mueff / Nvar) / (Nvar + 4 + 2 * mueff / Nvar);
  mucov = mueff;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  double t1 = 2. / ((Nvar + 1.4142) * (Nvar + 1.4142));
  double t2 = (2. * mueff - 1.) / ((Nvar + 2.) * (Nvar + 2.) + mueff);
  t2 = (t2 > 1) ? 1 : t2;
  t2 = (1. / mucov) * t1 + (1. - 1. / mucov) * t2;
  ccov = t2;
  damps = 1. +
          2 * (std::max(0., (std::sqrt((mueff - 1.) / (Nvar + 1.)) - 1.))) + cs;
  // diagonalCov = 2 + 100. * Nvar / sqrt((double) lambda);
}

// 段落说明：实现 `MultiMet::CMAES_initial`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::CMAES_initial() {
  sigma = 1.0;
  chiN = sqrt((double)Nvar);

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  pcc = new double[Nvar];
  ps = new double[Nvar];
  tempRandom = new double[Nvar + 1];
  BDz = new double[Nvar];
  xmean = new double[Nvar];
  xold = new double[Nvar];
  rgD = new double[Nvar];
  index = new int[lambda];

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < lambda; ++i)
    index[i] = i;

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int i = 0; i < Nvar; ++i) {
    xmean[i] = gbest[i];
    xold[i] = xmean[i];
    rgD[i] = stddev[i];
    pcc[i] = 0.0;
    ps[i] = 0.0;
    BDz[i] = 0.0;
    tempRandom[i] = 0.0;
  }
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
void MultiMet::CMAES_sampleGenerate(int firstrun, int p_start, int p_end) {
  (void)firstrun;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int iPop = p_start; iPop < p_end; ++iPop) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; ++j) {
      double step = sigma * rgD[j] * gauss();
      newpop[iPop][j] = xmean[j] + step;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (newpop[iPop][j] > Ubound)
        newpop[iPop][j] = Ubound;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (newpop[iPop][j] < Lbound)
        newpop[iPop][j] = Lbound;
    }
  }
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
double* MultiMet::CMAES_updateDistribution(int gen) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (gen <= 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return xmean;

  // 段落说明：排序或定位最优/最差元素，为选择、统计或更新提供确定顺序。
  sortIndex(pop_fit, index, lambda);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Nvar; ++i) {
    xold[i] = xmean[i];
    xmean[i] = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k < mu; ++k)
      xmean[i] += weights[k] * pop[index[k]][i];

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    double variance = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k < mu; ++k) {
      double diff = pop[index[k]][i] - xold[i];
      variance += weights[k] * diff * diff;
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    double old_var = rgD[i] * rgD[i];
    double next_var = 0.85 * old_var + 0.15 * variance;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (next_var < 1e-6)
      next_var = 1e-6;
    rgD[i] = sqrt(next_var);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (rgD[i] > 0.5)
      rgD[i] = 0.5;
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  sigma *= 0.995;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (sigma < 0.05)
    sigma = 0.05;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  else if (sigma > 1.0)
    sigma = 1.0;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  return xmean;
}

/**
 * Dirty index sort.
 */
void MultiMet::sortIndex(const double* rgFunVal, int* iindex, int n) {
  int i, j;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = 1, iindex[0] = 0; i < n; ++i) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (j = i; j > 0; --j) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (rgFunVal[iindex[j - 1]] < rgFunVal[i])
        break;
      iindex[j] = iindex[j - 1]; // shift up
    }
    iindex[j] = i;
  }
}
void MultiMet::adaptC2(const int hsig, int gen) {
  const int N = Nvar;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ccov != 0.) {
    // definitions for speeding up inner-most loop
    const double mucovinv = double(1) / mucov;
    const double commonFactor =
        ccov * ((gen == 0) ? (N + double(1.5)) / double(3) : double(1));
    const double ccov1 = std::min(commonFactor * mucovinv, double(1));
    const double ccovmu =
        std::min(commonFactor * (double(1) - mucovinv), double(1) - ccov1);
    const double sigmasquare = sigma * sigma;
    const double onemccov1ccovmu = double(1) - ccov1 - ccovmu;
    const double longFactor =
        (double(1) - hsig) * ccumcov * (double(2) - ccumcov);

    // update covariance matrix
    for (int i = 0; i < N; ++i)
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = (gen == 0) ? i : 0; j <= i; ++j) {
        double& Cij = C[i][j];
        Cij = onemccov1ccovmu * Cij +
              ccov1 * (pcc[i] * pcc[j] + longFactor * Cij);
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int k = 0; k < mu; ++k) { // additional rank mu update
          const double* rgrgxindexk = pop[index[k]];
          Cij += ccovmu * weights[k] * (rgrgxindexk[i] - xold[i]) *
                 (rgrgxindexk[j] - xold[j]) / sigmasquare;
        }
      }
    //// update maximal and minimal diagonal value
    // maxdiagC = mindiagC = C[0][0];
    // for(int i = 1; i < N; ++i)
    //{
    //   const double& Cii = C[i][i];
    //   if(maxdiagC < Cii)
    //     maxdiagC = Cii;
    //   else if(mindiagC > Cii)
    //     mindiagC = Cii;
    // }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::updateEigensystem() {
  eigen(rgD, B, tempRandom);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Nvar; ++i)
    rgD[i] = std::sqrt(rgD[i]);
}

/**
 * Calculating eigenvalues and vectors.
 * @param rgtmp (input) N+1-dimensional vector for temporal use.
 * @param diag (output) N eigenvalues.
 * @param Q (output) Columns are normalized eigenvectors.
 */
void MultiMet::eigen(double* diag, double** Q, double* rgtmp) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (C != Q) // copy C to Q
  {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < Nvar; ++i)
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j <= i; ++j)
        Q[i][j] = Q[j][i] = C[i][j];
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  householder(Q, diag, rgtmp);
  ql(diag, rgtmp, Q);
}

/**
 * Symmetric tridiagonal QL algorithm, iterative.
 * Computes the eigensystem from a tridiagonal matrix in roughtly 3N^3
 * operations code adapted from Java JAMA package, function tql2.
 * @param d input: Diagonale of tridiagonal matrix. output: eigenvalues.
 * @param e input: [1..n-1], off-diagonal, output from Householder
 * @param V input: matrix output of Householder. output: basis of
 *          eigenvectors, according to d
 */
void MultiMet::ql(double* d, double* e, double** V) {
  const int n = Nvar;
  double f(0);
  double tst1(0);
  const double eps(2.22e-16); // 2.0^-52.0 = 2.22e-16

  // shift input e
  double* ep1 = e;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (double *ep2 = e + 1, *const end = e + n; ep2 != end; ep1++, ep2++)
    *ep1 = *ep2;
  *ep1 = double(0); // never changed again

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int l = 0; l < n; l++) {
    // find small subdiagonal element
    double& el = e[l];
    double& dl = d[l];
    const double smallSDElement = std::fabs(dl) + std::fabs(el);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (tst1 < smallSDElement)
      tst1 = smallSDElement;
    const double epsTst1 = eps * tst1;
    int m = l;
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    while (m < n) {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (std::fabs(e[m]) <= epsTst1)
        break;
      m++;
    }

    // if m == l, d[l] is an eigenvalue, otherwise, iterate.
    if (m > l) {
      // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
      do {
        double h, g = dl;
        double& dl1r = d[l + 1];
        double p = (dl1r - g) / (2. * el);
        double r = myhypot(p, 1.);

        // compute implicit shift
        if (p < 0)
          r = -r;
        const double shift = p + r;
        dl = el / shift;
        h = g - dl;
        const double dl1 = el * shift;
        dl1r = dl1;
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int i = l + 2; i < n; i++)
          d[i] -= h;
        f += h;

        // implicit QL transformation.
        p = d[m];
        double c(1);
        double c2(1);
        double c3(1);
        const double el1 = e[l + 1];
        double s(0);
        double s2(0);
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int i = m - 1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          const double& ei = e[i];
          g = c * ei;
          h = c * p;
          r = myhypot(p, ei);
          e[i + 1] = s * r;
          s = ei / r;
          c = p / r;
          const double& di = d[i];
          p = c * di - s * g;
          d[i + 1] = h + s * (c * g + s * di);

          // accumulate transformation.
          for (int k = 0; k < n; k++) {
            double& Vki1 = V[k][i + 1];
            h = Vki1;
            double& Vki = V[k][i];
            Vki1 = s * Vki + c * h;
            Vki *= c;
            Vki -= s * h;
          }
        }
        p = -s * s2 * c3 * el1 * el / dl1;
        el = s * p;
        dl = c * p;
      } while (std::fabs(el) > epsTst1);
    }
    dl += f;
    el = 0.0;
  }
}
/**
 * Householder transformation of a symmetric matrix V into tridiagonal form.
 * Code slightly adapted from the Java JAMA package, function private tred2().
 * @param V input: symmetric nxn-matrix. output: orthogonal transformation
 *          matrix: tridiag matrix == V* V_in* V^double.
 * @param d output: diagonal
 * @param e output: [0..n-1], off diagonal (elements 1..n-1)
 */
void MultiMet::householder(double** V, double* d, double* e) {
  const int n = Nvar;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int j = 0; j < n; j++) {
    d[j] = V[n - 1][j];
  }

  // Householder reduction to tridiagonal form

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = n - 1; i > 0; i--) {
    // scale to avoid under/overflow
    double scale = 0.0;
    double h = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (double *pd = d, *const dend = d + i; pd != dend; pd++) {
      scale += std::fabs(*pd);
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (scale == 0.0) {
      e[i] = d[i - 1];
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < i; j++) {
        d[j] = V[i - 1][j];
        V[i][j] = 0.0;
        V[j][i] = 0.0;
      }
    } else {
      // generate Householder vector
      for (double *pd = d, *const dend = d + i; pd != dend; pd++) {
        *pd /= scale;
        h += *pd * *pd;
      }
      double& dim1 = d[i - 1];
      double f = dim1;
      double g = f > 0 ? -std::sqrt(h) : std::sqrt(h);
      e[i] = scale * g;
      h = h - f * g;
      dim1 = f - g;
      memset((void*)e, 0, (size_t)i * sizeof(double));

      // apply similarity transformation to remaining columns
      for (int j = 0; j < i; j++) {
        f = d[j];
        V[j][i] = f;
        double& ej = e[j];
        g = ej + V[j][j] * f;
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int k = j + 1; k <= i - 1; k++) {
          double& Vkj = V[k][j];
          g += Vkj * d[k];
          e[k] += Vkj * f;
        }
        ej = g;
      }
      f = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < i; j++) {
        double& ej = e[j];
        ej /= h;
        f += ej * d[j];
      }
      double hh = f / (h + h);
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < i; j++) {
        double& dj = d[j];
        f = dj;
        g = e[j];
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int k = j; k <= i - 1; k++) {
          V[k][j] -= f * e[k] + g * d[k];
        }
        dj = V[i - 1][j];
        V[i][j] = 0.0;
      }
    }
    d[i] = h;
  }

  // accumulate transformations
  const int nm1 = n - 1;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < nm1; i++) {
    double h;
    double& Vii = V[i][i];
    V[n - 1][i] = Vii;
    Vii = 1.0;
    h = d[i + 1];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (h != 0.0) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int k = 0; k <= i; k++) {
        d[k] = V[k][i + 1] / h;
      }
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j <= i; j++) {
        double g = 0.0;
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int k = 0; k <= i; k++) {
          double* Vk = V[k];
          g += Vk[i + 1] * Vk[j];
        }
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int k = 0; k <= i; k++) {
          V[k][j] -= g * d[k];
        }
      }
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k <= i; k++) {
      V[k][i + 1] = 0.0;
    }
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int j = 0; j < n; j++) {
    double& Vnm1j = V[n - 1][j];
    d[j] = Vnm1j;
    Vnm1j = 0.0;
  }
  V[n - 1][n - 1] = 1.0;
  e[0] = 0.0;
}

// 段落说明：排序或定位最优/最差元素，为选择、统计或更新提供确定顺序。
void MultiMet::BA(int p_start, int p_end) {
  (void)p_start;
  (void)p_end;
  pop_heap_sort(Popsize);
  newpop_heap_sort(Popsize);

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  for (int i = 0; i < ne; i++)
    NeighborFlowerPatch(nre, i);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (randval(0, 1) < 0.5)
    DE(randval(0.1, 0.9), rand() % 5, randval(0.1, 0.9), ne, Popsize);
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    mutate(1, ne, Popsize);
}

// 段落说明：实现 `MultiMet::NeighborFlowerPatch`：完成该函数负责的数据准备、算法步骤和状态返回。
void MultiMet::NeighborFlowerPatch(int nr, int point) {
  meme_selection(point, 0, ngh[point], nr);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  if (pop_fit[point] < newpop_fit[point]) {
    ngh[point] *= ngh_decay;
    ngh_decay_count[point]++;
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (ngh_decay_count[point] > stlim) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < Nvar; j++)
        newpop[point][j] = randval(Lbound, Ubound);
      newpop_fit[point] =
          EvaluFunc(newpop[point], Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                    CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
                    AvailDeviceList, EnergyList, CloudDevices, EdgeDevices,
                    CloudLoad, EdgeLoad, DeviceLoad, CETask_coDevice,
                    Edge_Device_comm, ST, ET, CE_ST, CE_ET);
      ngh[point] = ngh_origin;
      ngh_decay_count[point] = 0;
    }
  }
}

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
void MultiMet::Direct_Change(int ChangeSize) {
  double min_dis = 1e10;
  int min_index = -1;
  double tmp_dis = 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (delta_update_count > kArchiveSize) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int k = 0; k < ChangeSize; k++) {
      min_dis = 1e10;
      min_index = -1;
      int pop_ind = rand() % Popsize;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int i = 0; i < 10; i++) {
        tmp_dis = Euclidean_dis(pop[pop_ind], delta_pop[i], Nvar);
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (tmp_dis < min_dis) {
          min_dis = tmp_dis;
          min_index = i;
        }
      }
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (min_index >= 0 && min_index < 10) {
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (int i = 0; i < Nvar; i++) {
          newpop[pop_ind][i] += delta_pop[min_index][Nvar + i];
        }
      }
    }
  }
}

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include "MultimethodAde.cpp"
