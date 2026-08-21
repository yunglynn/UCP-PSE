/* CED loader, exact decoder/objective, detailed metrics, and lightweight hybrid
 * surrogate. Proxy screening never replaces the final exact audit. */
#include "Problems.h"
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstring>

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static constexpr double kCloudBackhaulRateDivisor = 5.0;
static constexpr double kMinimumWirelessDistanceKm = 0.01;
static constexpr int kProxyFeatureGroups = 4;
static constexpr int kProxyBinsPerGroup = 6;
static constexpr int kProxyFeatureCount =
    kProxyFeatureGroups * kProxyBinsPerGroup;
static constexpr int kProxyEnsembleSize = 3;
static CEDDetailedMetrics g_last_detailed_metrics;

// 段落说明：实现 `CED_LastDetailedMetrics`：完成该函数负责的数据准备、算法步骤和状态返回。
const CEDDetailedMetrics& CED_LastDetailedMetrics() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return g_last_detailed_metrics;
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
template <typename T>
static CEDLoadDistribution SummarizeLoads(const vector<T>& loads) {
  CEDLoadDistribution result;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (loads.empty())
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return result;
  vector<T> ordered(loads);
  sort(ordered.begin(), ordered.end());
  long double sum = 0.0L;
  long double square_sum = 0.0L;
  size_t used_count = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (T value : loads) {
    sum += value;
    square_sum += static_cast<long double>(value) * value;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (value > static_cast<T>(0))
      ++used_count;
  }
  result.mean = static_cast<double>(sum / loads.size());
  long double variance = 0.0L;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (T value : loads) {
    const long double difference = value - result.mean;
    variance += difference * difference;
  }
  result.standard_deviation =
      sqrt(static_cast<double>(variance / loads.size()));
  result.minimum = ordered.front();
  result.maximum = ordered.back();
  result.median = ordered[(ordered.size() - 1) / 2];
  result.percentile95 =
      ordered[static_cast<size_t>(ceil(0.95 * ordered.size())) - 1];
  result.jain_index = square_sum > 0.0L
                          ? static_cast<double>(sum * sum /
                                                (loads.size() * square_sum))
                          : 1.0;
  result.total_count = static_cast<double>(loads.size());
  result.used_count = static_cast<double>(used_count);
  result.active_jain_index =
      used_count > 0 && square_sum > 0.0L
          ? static_cast<double>(sum * sum / (used_count * square_sum))
          : 0.0;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return result;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static int ProxyResidualStumpCount() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return static_cast<int>(std::ceil(std::log2(kProxyEnsembleSize)));
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static int ProxyInitialDesignSize() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return static_cast<int>(
      std::floor(std::pow(static_cast<double>(kProxyFeatureCount), 2.0 / 3.0)));
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static int ProxyArchiveSize() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return kProxyFeatureCount;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static int ProxyRankWindow() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 2 * ProxyInitialDesignSize();
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static int ProxyFeasibleAnchorLimit() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return kProxyFeatureCount - ProxyInitialDesignSize() / 2;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static int ProxyPeriodicCheckInterval() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return ProxyFeasibleAnchorLimit() * (ProxyInitialDesignSize() / 2);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyKernelBandwidth() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 1.0 / static_cast<double>(kProxyFeatureCount);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyDistanceInflation() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return static_cast<double>(kProxyFeatureCount) /
         static_cast<double>(kProxyFeatureGroups + kProxyBinsPerGroup);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyOptimismMargin() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 1.0 / static_cast<double>(kProxyFeatureCount + kProxyFeatureGroups +
                                   kProxyBinsPerGroup);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyHighErrorBoundary() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return sqrt(static_cast<double>(kProxyEnsembleSize) /
              static_cast<double>(kProxyFeatureCount));
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyLowErrorBoundary() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 1.0 / sqrt(static_cast<double>(kProxyFeatureCount));
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyPoorPredictionMargin() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 1.0 / static_cast<double>(ProxyRankWindow() + 1);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyConservativeMargin() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 2.0 / static_cast<double>(ProxyRankWindow() + 1);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyGuardScale() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 2.0 / static_cast<double>(kProxyFeatureCount + 1);
}
static DistanceValue** cached_EtoD = nullptr;
static int cached_Enum = 0;
static int cached_Dnum = 0;
static vector<int> cached_nearest_edge;
static CETask* cached_task_property = nullptr;
static double* cached_mtask_time = nullptr;
static double* cached_energy_list = nullptr;
static int cached_scale_Cnum = 0;
static int cached_scale_Enum = 0;
static int cached_scale_Dnum = 0;
static int cached_scale_CE_Tnum = 0;
static int cached_scale_M_Jnum = 0;
static int cached_scale_M_OPTnum = 0;
static double cached_time_scale = 1.0;
static double cached_energy_scale = 1.0;
static const double* compact_edge_x = nullptr;
static const double* compact_edge_y = nullptr;
static const double* compact_device_x = nullptr;
static const double* compact_device_y = nullptr;
static int compact_edge_count = 0;
static int compact_device_count = 0;

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
void CED_SetCompactGeometry(const double* edge_x, const double* edge_y,
                            const double* device_x, const double* device_y,
                            int edge_count, int device_count) {
  compact_edge_x = edge_x;
  compact_edge_y = edge_y;
  compact_device_x = device_x;
  compact_device_y = device_y;
  compact_edge_count = edge_count;
  compact_device_count = device_count;
  cached_EtoD = reinterpret_cast<DistanceValue**>(-1);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static DistanceValue CompactEncodedDistance(double dx, double dy) {
  const double scaled = std::hypot(dx, dy) / DISTANCE_SCALE;
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

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static DistanceValue EdgeDeviceDistance(DistanceValue** matrix, int edge,
                                        int device) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (matrix != nullptr)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return matrix[edge][device];
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (edge < 0 || edge >= compact_edge_count || device < 0 ||
      device >= compact_device_count)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return CompactEncodedDistance(compact_edge_x[edge] - compact_device_x[device],
                                compact_edge_y[edge] - compact_device_y[device]);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static DistanceValue DeviceDistance(DistanceValue** matrix, int first,
                                    int second) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (matrix != nullptr)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return matrix[first][second];
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (first < 0 || first >= compact_device_count || second < 0 ||
      second >= compact_device_count)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return CompactEncodedDistance(
      compact_device_x[first] - compact_device_x[second],
      compact_device_y[first] - compact_device_y[second]);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static double UnitValue(double value) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(value))
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.5;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value < 0.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value > 1.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1.0;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return value;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static int UnitIndex(double value, int count) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (count <= 1)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  int index = (int)(UnitValue(value) * (count - 1));
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (index < 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (index >= count)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return count - 1;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return index;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static double ClampRate(double rate) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(rate) || rate < 1e-6)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1e-6;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (rate > 10000.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 10000.0;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return rate;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static double LinkRate(DistanceValue distance_value, double interference_gain) {
  double distance_km = DecodeDistance(distance_value) / 1000.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(distance_km) ||
      distance_km < kMinimumWirelessDistanceKm)
    distance_km = kMinimumWirelessDistanceKm;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  double current_gain = kTransmitPowerDbm / distance_km;
  double denominator = fabs(interference_gain);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(denominator) || denominator < 1.0)
    denominator = 1.0;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  return ClampRate(20.0 * log2(1.0 + current_gain / denominator) / 8.0);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static double LoadPenalty(int load, int capacity) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (capacity <= 0 || load <= capacity)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1.0;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return (double)load / (double)capacity;
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static void SortUniqueDevices(vector<int>& devices) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (devices.size() < 2)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  sort(devices.begin(), devices.end());
  devices.erase(unique(devices.begin(), devices.end()), devices.end());
}

// Quantize normalized scheduling priorities to 16-bit rank keys. The stable
// counting pass preserves index order for equal keys and avoids O(n log n)
// comparison sorting.
static void SortUnitKeyIndices(vector<int>& indices,
                               const vector<double>& keys,
                               vector<int>& scratch) {
  const size_t n = indices.size();
  scratch.resize(n);
  static vector<size_t> count(65536);
  static vector<size_t> offset(65536);
  std::fill(count.begin(), count.end(), 0);
  auto rank_key = [&](int index) {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return static_cast<unsigned>(UnitValue(keys[index]) * 65535.0);
  };
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int index : indices)
    ++count[rank_key(index)];
  offset[0] = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int bucket = 1; bucket < 65536; ++bucket)
    offset[bucket] = offset[bucket - 1] + count[bucket - 1];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int index : indices)
    scratch[offset[rank_key(index)]++] = index;
  indices.swap(scratch);
}

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
static void EnsureDistanceCache(int Enum, int Dnum,
                                DistanceValue** EtoD_Distance) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (cached_EtoD == EtoD_Distance && cached_Enum == Enum &&
      cached_Dnum == Dnum)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  cached_EtoD = EtoD_Distance;
  cached_Enum = Enum;
  cached_Dnum = Dnum;
  cached_nearest_edge.assign(Dnum, 0);

  // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
  for (int i = 0; i < Dnum; i++) {
    double min_dis = 1e100;
    int min_index = 0;
    int edge_begin = 0;
    int edge_end = Enum;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (EtoD_Distance == nullptr && Enum * 16 == Dnum) {
      const int factory = i / 64;
      edge_begin = factory * 4;
      edge_end = edge_begin + 4;
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = edge_begin; j < edge_end; j++) {
      double current_dis = DecodeDistance(EdgeDeviceDistance(EtoD_Distance, j, i));
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (min_dis > current_dis) {
        min_dis = current_dis;
        min_index = j;
      }
    }
    cached_nearest_edge[i] = min_index;
  }
}
static void EnsureObjectiveScale(int Cnum, int Enum, int Dnum, int CE_Tnum,
                                 int M_Jnum, int M_OPTnum,
                                 CETask* CETask_Property, double* MTask_Time,
                                 double* EnergyList) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (cached_task_property == CETask_Property &&
      cached_mtask_time == MTask_Time && cached_energy_list == EnergyList &&
      cached_scale_Cnum == Cnum && cached_scale_Enum == Enum &&
      cached_scale_Dnum == Dnum && cached_scale_CE_Tnum == CE_Tnum &&
      cached_scale_M_Jnum == M_Jnum && cached_scale_M_OPTnum == M_OPTnum)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  cached_task_property = CETask_Property;
  cached_mtask_time = MTask_Time;
  cached_energy_list = EnergyList;
  cached_scale_Cnum = Cnum;
  cached_scale_Enum = Enum;
  cached_scale_Dnum = Dnum;
  cached_scale_CE_Tnum = CE_Tnum;
  cached_scale_M_Jnum = M_Jnum;
  cached_scale_M_OPTnum = M_OPTnum;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double mtask_sum = 0.0;
  double max_job_chain = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int job = 0; job < M_Jnum; job++) {
    double job_chain = 0.0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int operation = 0; operation < M_OPTnum; operation++)
      job_chain += MTask_Time[job * M_OPTnum + operation];
    mtask_sum += job_chain;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (job_chain > max_job_chain)
      max_job_chain = job_chain;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double compute_work = 0.0;
  double communication_sum = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < CE_Tnum; i++) {
    compute_work += CETask_Property[i].Computation;
    communication_sum += CETask_Property[i].Communication;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  vector<double> critical_path(CE_Tnum, 0.0);
  double max_compute_path = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < CE_Tnum; i++) {
    double predecessor_path = 0.0;
    const vector<int>* relations[] = {
        &CETask_Property[i].Precedence, &CETask_Property[i].Interact,
        &CETask_Property[i].Start_Pre, &CETask_Property[i].End_Pre};
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (const vector<int>* relation : relations) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int predecessor : *relation) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (predecessor >= 0 && predecessor < i &&
            critical_path[predecessor] > predecessor_path)
          predecessor_path = critical_path[predecessor];
      }
    }
    const double fastest_compute_time =
        CETask_Property[i].Computation / 3.7;
    const double nominal_communication_time =
        CETask_Property[i].Communication / 1000.0;
    critical_path[i] = predecessor_path + fastest_compute_time +
                       nominal_communication_time;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (critical_path[i] > max_compute_path)
      max_compute_path = critical_path[i];
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double max_energy_rate = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < 11; i++)
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (max_energy_rate < EnergyList[i])
      max_energy_rate = EnergyList[i];

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  const double manufacturing_capacity_bound =
      mtask_sum / static_cast<double>(max(1, Dnum));
  const double aggregate_compute_rate =
      3.7 * static_cast<double>(max(0, Cnum)) +
      2.2 * static_cast<double>(max(0, Enum));
  const double compute_capacity_bound =
      compute_work / max(1.0, aggregate_compute_rate);
  cached_time_scale =
      max(max(manufacturing_capacity_bound, max_job_chain),
          max(compute_capacity_bound, max_compute_path));
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (cached_time_scale <= 1e-12)
    cached_time_scale = 1.0;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  const int active_compute_nodes = min(CE_Tnum, max(1, Cnum + Enum));
  cached_energy_scale =
      max_energy_rate * cached_time_scale * active_compute_nodes / 1000.0 +
                        communication_sum * kTransmitPowerDbm / 1000.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (cached_energy_scale <= 1e-12)
    cached_energy_scale = 1.0;
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
double CED_Schedule(double* var, int Cnum, int Enum, int Dnum, int CE_Tnum,
                    int M_Jnum, int M_OPTnum, CETask* CETask_Property,
                    double* MTask_Time, DistanceValue** EtoD_Distance,
                    DistanceValue** DtoD_Distance, vector<int>* AvailDeviceList,
                    double* EnergyList, vector<int>* CloudDevices,
                    vector<int>* EdgeDevices, vector<int>* CloudLoad,
                    vector<int>* EdgeLoad, vector<int>*,
                    vector<int>* CETask_coDevice, map<int, double>*,
                    double** ST, double** ET, double* CE_ST, double* CE_ET) {
  (void)CETask_coDevice;
  const int opt_count = M_Jnum * M_OPTnum;
  static vector<unsigned char> ce_sele_storage;
  static vector<int> cevar_storage;
  static vector<int> mvar_storage;
  static vector<int> order_storage;
  static vector<int> order_scratch_storage;
  static vector<double> order_key_storage;
  static vector<int> geneO_storage;
  static vector<int> last_device_op_storage;
  static vector<int> co_device_storage;
  static vector<unsigned char> co_device_count_storage;
  static vector<double> edge_smallest_rate_storage;
  static vector<double> edge_comm_storage;

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  ce_sele_storage.resize(CE_Tnum);
  cevar_storage.resize(CE_Tnum);
  mvar_storage.resize(opt_count);
  order_storage.resize(opt_count);
  order_key_storage.resize(opt_count);
  geneO_storage.assign(M_Jnum, -1);
  last_device_op_storage.assign(Dnum, -1);
  co_device_storage.resize(CE_Tnum * M_OPTnum);
  co_device_count_storage.assign(CE_Tnum, 0);
  edge_smallest_rate_storage.assign(Enum, 1e10);
  const bool factory_local_devices = Enum > 0 && Enum * 16 == Dnum;
  const int edge_comm_stride = factory_local_devices ? 64 : Dnum;
  const size_t edge_comm_size = static_cast<size_t>(Enum) * edge_comm_stride;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (edge_comm_storage.size() < edge_comm_size)
    edge_comm_storage.resize(edge_comm_size);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  unsigned char* ce_sele = ce_sele_storage.data();
  int* cevar = cevar_storage.data();
  int* mvar = mvar_storage.data();
  int* seq_var = order_storage.data();
  double* order_key = order_key_storage.data();
  int* geneO = geneO_storage.data();
  int* last_device_op = last_device_op_storage.data();
  int* co_device = co_device_storage.data();
  unsigned char* co_device_count = co_device_count_storage.data();
  double* Edge_smallest_rate = edge_smallest_rate_storage.data();
  double* edge_comm = edge_comm_storage.data();

  // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
  EnsureDistanceCache(Enum, Dnum, EtoD_Distance);
  EnsureObjectiveScale(Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
                       CETask_Property, MTask_Time, EnergyList);
  const vector<int>& nearest_edge = cached_nearest_edge;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Cnum; i++)
    CloudDevices[i].clear();
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Enum; i++)
    EdgeDevices[i].clear();
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Cnum; i++)
    CloudLoad[i].clear();
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Enum; i++)
    EdgeLoad[i].clear();

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  for (int i = 0; i < CE_Tnum; i++) {
    ce_sele[i] = (UnitValue(var[i]) > 0.5) ? 1 : 0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (ce_sele[i] == false)
      cevar[i] = UnitIndex(var[CE_Tnum + i], Cnum);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else {
      int edge_count = (int)CETask_Property[i].AvailEdgeServerList.size();
      // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
      if (edge_count <= 0)
        // 控制说明：返回本阶段计算结果或状态码给调用方。
        return 1e300;
      cevar[i] = UnitIndex(var[CE_Tnum + i], edge_count);
      cevar[i] = CETask_Property[i].AvailEdgeServerList[cevar[i]];
    }
  }
  const int seq_gene_base = 2 * CE_Tnum;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < opt_count; i++) {
    order_storage[i] = i;
    order_key[i] = UnitValue(var[seq_gene_base + i]);
  }
  SortUnitKeyIndices(order_storage, order_key_storage,
                     order_scratch_storage);

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  for (int i = 0; i < opt_count; i++) {
    const int operation_index = seq_var[i];
    int device_count = (int)AvailDeviceList[operation_index].size();
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (device_count <= 0)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return 1e300;
    mvar[operation_index] =
        UnitIndex(var[CE_Tnum * 2 + M_Jnum * M_OPTnum + operation_index],
                  device_count);
    mvar[operation_index] =
        AvailDeviceList[operation_index][mvar[operation_index]];
  }

  // 段落说明：根据搜索状态选择动作并用改进收益更新策略统计量。
  int current_job;
  int current_machine;
  int current_operation;
  int previous_operation;
  int previous_machine_operation;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < opt_count; i++) {
    current_job = seq_var[i] / M_OPTnum;
    geneO[current_job]++;
    current_operation = geneO[current_job];
    current_machine = mvar[seq_var[i]];
    previous_operation = current_operation - 1;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (previous_operation < 0) // no precedence
    {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (last_device_op[current_machine] <
          0) // target machine has no operation
      {
        ST[current_job][current_operation] = 0;
        ET[current_job][current_operation] =
            MTask_Time[current_job * M_OPTnum + current_operation];
      } else {
        previous_machine_operation = last_device_op[current_machine];
        ST[current_job][current_operation] =
            ET[previous_machine_operation / M_OPTnum]
              [previous_machine_operation % M_OPTnum];
        ET[current_job][current_operation] =
            ST[current_job][current_operation] +
            MTask_Time[current_job * M_OPTnum + current_operation];
      }
      last_device_op[current_machine] =
          current_job * M_OPTnum + current_operation;
    } else // 有前驱工序
    {
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (last_device_op[current_machine] < 0) {
        ST[current_job][current_operation] =
            ET[current_job][previous_operation];
        ET[current_job][current_operation] =
            ST[current_job][current_operation] +
            MTask_Time[current_job * M_OPTnum + current_operation] +
            DecodeDistance(DeviceDistance(
                DtoD_Distance,
                mvar[current_job * M_OPTnum + previous_operation],
                mvar[current_job * M_OPTnum + current_operation])) /
                (100000.0 / 3600.0);
      } else {
        previous_machine_operation = last_device_op[current_machine];
        ST[current_job][current_operation] =
            fmax(ET[current_job][previous_operation],
                 ET[previous_machine_operation / M_OPTnum]
                   [previous_machine_operation % M_OPTnum]);
        ET[current_job][current_operation] =
            ST[current_job][current_operation] +
            MTask_Time[current_job * M_OPTnum + current_operation] +
            DecodeDistance(DeviceDistance(
                DtoD_Distance,
                mvar[current_job * M_OPTnum + previous_operation],
                mvar[current_job * M_OPTnum + current_operation])) /
                (100000.0 / 3600.0);
      }
      last_device_op[current_machine] =
          current_job * M_OPTnum + current_operation;
    }
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double total_operation_queue_wait = 0.0;
  double total_transportation_time = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int job = 0; job < M_Jnum; ++job) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int operation = 0; operation < M_OPTnum; ++operation) {
      const double ready_time =
          operation == 0 ? 0.0 : ET[job][operation - 1];
      total_operation_queue_wait +=
          max(0.0, ST[job][operation] - ready_time);
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (operation > 0) {
        const int previous_device = mvar[job * M_OPTnum + operation - 1];
        const int current_device = mvar[job * M_OPTnum + operation];
        total_transportation_time +=
            DecodeDistance(DeviceDistance(DtoD_Distance, previous_device,
                                          current_device)) /
            (100000.0 / 3600.0);
      }
    }
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < CE_Tnum; i++) {
    int base = i * M_OPTnum;
    unsigned char count = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < M_OPTnum; j++) {
      int device = mvar[base + j];
      bool seen = false;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int k = 0; k < count; k++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (co_device[base + k] == device) {
          seen = true;
          break;
        }
      }
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (!seen)
        co_device[base + count++] = device;
    }
    co_device_count[i] = count;
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < CE_Tnum; i++) {
    const int task_job = i;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (ce_sele[i] == false) // cloud mode
    {
      int base = task_job * M_OPTnum;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < co_device_count[task_job]; j++) {
        int device = co_device[base + j];
        CloudDevices[cevar[i]].push_back(device);

        // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
        int access_edge = nearest_edge[device];
        EdgeDevices[access_edge].push_back(device);
      }
      CloudLoad[cevar[i]].push_back(i);
    } else {
      int base = task_job * M_OPTnum;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < co_device_count[task_job]; j++) {
        EdgeDevices[cevar[i]].push_back(co_device[base + j]);
      }
      EdgeLoad[cevar[i]].push_back(i);
    }
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Cnum; i++)
    SortUniqueDevices(CloudDevices[i]);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Enum; i++)
    SortUniqueDevices(EdgeDevices[i]);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Enum; i++) {
    double bottom_sum = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (auto iter = EdgeDevices[i].begin(); iter != EdgeDevices[i].end();
         iter++) {
      double distance_km =
          DecodeDistance(EdgeDeviceDistance(EtoD_Distance, i, *iter)) / 1000.0;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (!std::isfinite(distance_km) ||
          distance_km < kMinimumWirelessDistanceKm)
        distance_km = kMinimumWirelessDistanceKm;
      bottom_sum += kTransmitPowerDbm / distance_km;
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (auto iter = EdgeDevices[i].begin(); iter != EdgeDevices[i].end();
         iter++) {
      double distance_km =
          DecodeDistance(EdgeDeviceDistance(EtoD_Distance, i, *iter)) / 1000.0;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (!std::isfinite(distance_km) ||
          distance_km < kMinimumWirelessDistanceKm)
        distance_km = kMinimumWirelessDistanceKm;
      double current_gain = kTransmitPowerDbm / distance_km;
      double transmission_rate =
          LinkRate(EdgeDeviceDistance(EtoD_Distance, i, *iter),
                   bottom_sum - current_gain - 100.0);
      const int local_device = factory_local_devices ? *iter % 64 : *iter;
      edge_comm[static_cast<size_t>(i) * edge_comm_stride + local_device] =
          transmission_rate;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (transmission_rate < Edge_smallest_rate[i])
        Edge_smallest_rate[i] = transmission_rate;
    }
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < CE_Tnum; i++)
    CE_ST[i] = CE_ET[i] = 0;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double time_max = 0;
  double energy = 0;
  double total_task_communication_time = 0.0;
  vector<double> cloud_busy_time(Cnum, 0.0);
  vector<double> edge_busy_time(Enum, 0.0);
  vector<double> device_busy_time(Dnum, 0.0);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int operation = 0; operation < opt_count; ++operation) {
    const int device = mvar[operation];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (device >= 0 && device < Dnum)
      device_busy_time[device] += MTask_Time[operation];
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < CE_Tnum; i++) {
    double t_comm = 0, t_comp = 0;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (ce_sele[i] == false) // cloud mode
    {
      int base = i * M_OPTnum;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < co_device_count[i]; j++) {
        int device = co_device[base + j];
        int edge_index = nearest_edge[device];
        double link_rate = Edge_smallest_rate[edge_index];
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (!std::isfinite(link_rate) || link_rate > 1e9)
          link_rate =
              LinkRate(EdgeDeviceDistance(EtoD_Distance, edge_index, device),
                       -100.0);
        link_rate = ClampRate(link_rate / kCloudBackhaulRateDivisor);
        double cur_comm =
            CETask_Property[i].Communication * 10 / (1000 * link_rate);
        energy += cur_comm * kTransmitPowerDbm /
                  1000; // offloading transmission energy
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (cur_comm > t_comm)
          t_comm = cur_comm;
      }
    } else {
      int base = i * M_OPTnum;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int j = 0; j < co_device_count[i]; j++) {
        int device = co_device[base + j];
        const int local_device = factory_local_devices ? device % 64 : device;
        double link_rate = ClampRate(
            edge_comm[static_cast<size_t>(cevar[i]) * edge_comm_stride +
                      local_device]);
        double cur_comm =
            CETask_Property[i].Communication * 10 / (1000 * link_rate);
        energy += cur_comm * kTransmitPowerDbm /
                  1000; // offloading transmission energy
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (cur_comm > t_comm)
          t_comm = cur_comm;
      }
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (ce_sele[i] == false)
      t_comp = CETask_Property[i].Computation / 3.7 *
               LoadPenalty((int)CloudLoad[cevar[i]].size(), 20);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else {
      t_comp = CETask_Property[i].Computation / 2.2 *
               LoadPenalty((int)EdgeLoad[cevar[i]].size(), 6);
    }
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (ce_sele[i] == false && cevar[i] >= 0 && cevar[i] < Cnum)
      cloud_busy_time[cevar[i]] += t_comp;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (ce_sele[i] == true && cevar[i] >= 0 && cevar[i] < Enum)
      edge_busy_time[cevar[i]] += t_comp;

    // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
    total_task_communication_time += t_comm;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    double max_Prec_EndTime = 0, max_Start_StartTime = 0, max_End_EndTime = 0,
           max_Iter_EndTime = 0;
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (CETask_Property[i].Precedence.size() != 0) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (auto iter = CETask_Property[i].Precedence.begin();
           iter != CETask_Property[i].Precedence.end(); iter++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (max_Prec_EndTime < CE_ET[*iter])
          max_Prec_EndTime = CE_ET[*iter];
      }
    }
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (CETask_Property[i].Start_Pre.size() != 0) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (auto iter = CETask_Property[i].Start_Pre.begin();
           iter != CETask_Property[i].Start_Pre.end(); iter++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (max_Start_StartTime < CE_ST[*iter])
          max_Start_StartTime = CE_ST[*iter];
      }
    }
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (CETask_Property[i].End_Pre.size() != 0) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (auto iter = CETask_Property[i].End_Pre.begin();
           iter != CETask_Property[i].End_Pre.end(); iter++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (max_End_EndTime < CE_ET[*iter])
          max_End_EndTime = CE_ET[*iter];
      }
    }
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (CETask_Property[i].Interact.size() != 0) {
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (auto iter = CETask_Property[i].Interact.begin();
           iter != CETask_Property[i].Interact.end(); iter++) {
        // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
        if (max_Iter_EndTime < CE_ET[*iter])
          max_Iter_EndTime = CE_ET[*iter];
      }
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    const int task_job = i;
    CE_ST[i] = fmax(max_Start_StartTime, max_Prec_EndTime);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (CETask_Property[i].Job_Constraints == 1 ||
        CETask_Property[i].Job_Constraints == 3)
      CE_ST[i] = fmax(CE_ST[i], ST[task_job][M_OPTnum - 1]);
    CE_ET[i] = CE_ST[i] + t_comm + t_comp;
    CE_ET[i] = fmax(CE_ET[i], max_End_EndTime);
    CE_ET[i] = fmax(CE_ET[i], max_Iter_EndTime);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (CETask_Property[i].Job_Constraints == 2 ||
        CETask_Property[i].Job_Constraints == 3)
      CE_ET[i] = fmax(CE_ET[i], ET[task_job][M_OPTnum - 1]);

    // update the endtime of the interact task
    if (CETask_Property[i].Interact.size() != 0)
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (auto iter = CETask_Property[i].Interact.begin();
           iter != CETask_Property[i].Interact.end(); iter++)
        CE_ET[*iter] = CE_ET[i];
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < CE_Tnum; i++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (time_max < CE_ET[i])
      time_max = CE_ET[i];
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Cnum; i++) {
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (CloudLoad[i].size() == 0)
      continue;
    int u_ratio = static_cast<int>(
        (static_cast<double>(CloudLoad[i].size()) / 20.0) * 10.0);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (u_ratio > 10)
      u_ratio = 10;
    double time_expand = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (auto iter = CloudLoad[i].begin(); iter != CloudLoad[i].end(); iter++)
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (CE_ET[*iter] - CE_ST[*iter] > time_expand)
        time_expand = CE_ET[*iter] - CE_ST[*iter];
    energy += EnergyList[u_ratio] * time_expand / 1000.0;
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Enum; i++) {
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (EdgeLoad[i].size() == 0)
      continue;
    int u_ratio = static_cast<int>(
        (static_cast<double>(EdgeLoad[i].size()) / 6.0) * 10.0);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (u_ratio > 10)
      u_ratio = 10;
    double time_expand = 0;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (auto iter = EdgeLoad[i].begin(); iter != EdgeLoad[i].end(); iter++)
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (CE_ET[*iter] - CE_ST[*iter] > time_expand)
        time_expand = CE_ET[*iter] - CE_ST[*iter];
    energy += EnergyList[u_ratio] * time_expand / 1000.0;
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  if (!std::isfinite(time_max) || !std::isfinite(energy) || time_max < 0.0 ||
      energy < 0.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1e300;

  // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
  double normalized_time = time_max / cached_time_scale;
  double normalized_energy = energy / cached_energy_scale;
  double objective = normalized_time + normalized_energy;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (!std::isfinite(objective))
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1e300;

  // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
  vector<int> cloud_task_load(Cnum, 0);
  vector<int> edge_task_load(Enum, 0);
  vector<int> industrial_operation_load(Dnum, 0);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int node = 0; node < Cnum; ++node)
    cloud_task_load[node] = static_cast<int>(CloudLoad[node].size());
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int node = 0; node < Enum; ++node)
    edge_task_load[node] = static_cast<int>(EdgeLoad[node].size());
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int operation = 0; operation < opt_count; ++operation)
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (mvar[operation] >= 0 && mvar[operation] < Dnum)
      ++industrial_operation_load[mvar[operation]];
  g_last_detailed_metrics.objective = objective;
  g_last_detailed_metrics.energy = energy;
  g_last_detailed_metrics.makespan = time_max;
  g_last_detailed_metrics.average_operation_queue_wait =
      total_operation_queue_wait / static_cast<double>(max(1, opt_count));
  g_last_detailed_metrics.average_task_communication_time =
      total_task_communication_time /
      static_cast<double>(max(1, CE_Tnum));
  g_last_detailed_metrics.total_transportation_time =
      total_transportation_time;
  g_last_detailed_metrics.cloud_load = SummarizeLoads(cloud_task_load);
  g_last_detailed_metrics.edge_load = SummarizeLoads(edge_task_load);
  g_last_detailed_metrics.device_load =
      SummarizeLoads(industrial_operation_load);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const int factory_count = max(1, CE_Tnum / 1000);
  const int cloud_per_factory = max(1, Cnum / factory_count);
  const int edge_per_factory = max(1, Enum / factory_count);
  const int device_per_factory = max(1, Dnum / factory_count);
  vector<double> factory_busy_time(factory_count, 0.0);
  vector<double> cloud_idle_time(Cnum, 0.0);
  vector<double> edge_idle_time(Enum, 0.0);
  vector<double> device_idle_time(Dnum, 0.0);
  vector<double> all_resource_busy_time;
  vector<double> all_resource_idle_time;
  all_resource_busy_time.reserve(Cnum + Enum + Dnum);
  all_resource_idle_time.reserve(Cnum + Enum + Dnum);
  auto accumulate_resource_times =
      [&](const vector<double>& busy, vector<double>& idle,
          int resources_per_factory) {
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (size_t resource = 0; resource < busy.size(); ++resource) {
          idle[resource] = max(0.0, time_max - busy[resource]);
          all_resource_busy_time.push_back(busy[resource]);
          all_resource_idle_time.push_back(idle[resource]);
          const int factory = min(factory_count - 1,
                                  static_cast<int>(resource) /
                                      resources_per_factory);
          factory_busy_time[factory] += busy[resource];
        }
      };
  accumulate_resource_times(cloud_busy_time, cloud_idle_time,
                            cloud_per_factory);
  accumulate_resource_times(edge_busy_time, edge_idle_time, edge_per_factory);
  accumulate_resource_times(device_busy_time, device_idle_time,
                            device_per_factory);
  const double factory_capacity =
      static_cast<double>(cloud_per_factory + edge_per_factory +
                          device_per_factory) *
      time_max;
  vector<double> factory_idle_time(factory_count, 0.0);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int factory = 0; factory < factory_count; ++factory)
    factory_idle_time[factory] =
        max(0.0, factory_capacity - factory_busy_time[factory]);

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  g_last_detailed_metrics.factory_busy_time =
      SummarizeLoads(factory_busy_time);
  g_last_detailed_metrics.factory_idle_time =
      SummarizeLoads(factory_idle_time);
  g_last_detailed_metrics.all_resource_busy_time =
      SummarizeLoads(all_resource_busy_time);
  g_last_detailed_metrics.all_resource_idle_time =
      SummarizeLoads(all_resource_idle_time);
  g_last_detailed_metrics.cloud_busy_time = SummarizeLoads(cloud_busy_time);
  g_last_detailed_metrics.cloud_idle_time = SummarizeLoads(cloud_idle_time);
  g_last_detailed_metrics.edge_busy_time = SummarizeLoads(edge_busy_time);
  g_last_detailed_metrics.edge_idle_time = SummarizeLoads(edge_idle_time);
  g_last_detailed_metrics.device_busy_time = SummarizeLoads(device_busy_time);
  g_last_detailed_metrics.device_idle_time = SummarizeLoads(device_idle_time);

  // 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
  return objective;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
template <typename ValueAt>
static void
BuildKrigingProxyFeatureFromValue(int CE_Tnum, int M_Jnum, int M_OPTnum,
                                  vector<double>& feature, ValueAt value_at) {
  const int opt_count = M_Jnum * M_OPTnum;
  const int feature_count = kProxyFeatureCount;
  feature.assign(feature_count, 0.0);

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  int count[kProxyFeatureCount] = {0};
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < CE_Tnum; i += max(1, CE_Tnum / 256)) {
    double v0 = UnitValue(value_at(i));
    double v1 = UnitValue(value_at(CE_Tnum + i));
    int b0 = i % kProxyBinsPerGroup;
    int b1 = kProxyBinsPerGroup + i % kProxyBinsPerGroup;
    feature[b0] += v0;
    feature[b1] += v1;
    count[b0]++;
    count[b1]++;
  }

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  int seq_base = 2 * CE_Tnum;
  int machine_base = 2 * CE_Tnum + opt_count;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < opt_count; i += max(1, opt_count / 512)) {
    double v0 = UnitValue(value_at(seq_base + i));
    double v1 = UnitValue(value_at(machine_base + i));
    int b0 = 2 * kProxyBinsPerGroup + i % kProxyBinsPerGroup;
    int b1 = 3 * kProxyBinsPerGroup + i % kProxyBinsPerGroup;
    feature[b0] += v0;
    feature[b1] += v1;
    count[b0]++;
    count[b1]++;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < feature_count; i++) {
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (count[i] > 0)
      feature[i] /= count[i];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!std::isfinite(feature[i]))
      feature[i] = 0.5;
  }
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static void BuildKrigingProxyFeature(double* var, int CE_Tnum, int M_Jnum,
                                     int M_OPTnum, vector<double>& feature) {
  BuildKrigingProxyFeatureFromValue(CE_Tnum, M_Jnum, M_OPTnum, feature,
                                    [&](int index) { return var[index]; });
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double KrigingProxyDistance(const vector<double>& a,
                                   const vector<double>& b) {
  double dist = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (size_t i = 0; i < a.size(); i++) {
    double diff = a[i] - b[i];
    dist += diff * diff;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return dist;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct ClassifierProxyClass {
  vector<double> center;
  double mean;
  double best;
  double worst;
  int count;
};

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct ProxyStump {
  int feature;
  double threshold;
  double left_value;
  double right_value;
};

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static void UpdateClassifierProxyClass(ClassifierProxyClass& klass,
                                       const vector<double>& feature,
                                       double value) {
  klass.count++;
  double rate = 1.0 / klass.count;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (klass.center.empty())
    klass.center = feature;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (size_t i = 0; i < klass.center.size(); i++)
      klass.center[i] += rate * (feature[i] - klass.center[i]);
  }
  klass.mean += rate * (value - klass.mean);
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (value < klass.best)
    klass.best = value;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (value > klass.worst)
    klass.worst = value;
}

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
static double TrueScheduleValue(
    double* var, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum,
    int M_OPTnum, CETask* CETask_Property, double* MTask_Time,
    DistanceValue** EtoD_Distance, DistanceValue** DtoD_Distance,
    vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices,
    vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad,
    vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
    map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST,
    double* CE_ET) {
  double value = CED_Schedule(
      var, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
      MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
      CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
      CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return std::isfinite(value) ? value : 1e300;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct AdaptiveProxyState {
  double error;
  int eval_count;
  int true_eval_count;
  int exact_cache_hit_count;
  double best_true;
  vector<vector<double>> samples;
  vector<double> values;
  vector<double> residual_values;
  vector<ProxyStump> residual_stumps;
  int residual_model_sample_count;
  int residual_model_build_eval;
  vector<double> rank_proxy_values;
  vector<double> rank_true_values;
  vector<uint64_t> exact_true_hash_a;
  vector<uint64_t> exact_true_hash_b;
  vector<double> exact_true_values;
  double ensemble_loss[kProxyEnsembleSize];
  int ensemble_observations;
  ClassifierProxyClass good_class;
  ClassifierProxyClass bad_class;
  CETask* last_task;
  int last_cnum;
  int last_enum;
  int last_dnum;
  int last_tnum;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  AdaptiveProxyState()
      : eval_count(0), true_eval_count(0), exact_cache_hit_count(0),
        best_true(1e300),
        good_class{vector<double>(), 0.0, 1e300, 0.0, 0},
        bad_class{vector<double>(), 0.0, 1e300, 0.0, 0}, last_task(nullptr),
        last_cnum(0), last_enum(0), last_dnum(0), last_tnum(0) {
    Reset();
  }

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  void Reset() {
    error = 0.0;
    eval_count = 0;
    true_eval_count = 0;
    exact_cache_hit_count = 0;
    best_true = 1e300;
    samples.clear();
    values.clear();
    residual_values.clear();
    residual_stumps.clear();
    residual_model_sample_count = 0;
    residual_model_build_eval = 0;
    rank_proxy_values.clear();
    rank_true_values.clear();
    exact_true_hash_a.clear();
    exact_true_hash_b.clear();
    exact_true_values.clear();
    std::fill(ensemble_loss, ensemble_loss + kProxyEnsembleSize, 0.0);
    ensemble_observations = 0;
    good_class = {vector<double>(), 0.0, 1e300, 0.0, 0};
    bad_class = {vector<double>(), 0.0, 1e300, 0.0, 0};
  }
};

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static AdaptiveProxyState& SharedAdaptiveProxyState() {
  static AdaptiveProxyState state;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return state;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
double CED_ProxyBestTrueValue() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return SharedAdaptiveProxyState().best_true;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static bool g_proxy_policy_reduce_checks = false;
static double g_proxy_policy_strength = 0.0;
static bool g_proxy_policy_disagreement_audit = false;
static bool g_force_next_proxy_true_evaluation = false;
static double g_last_proxy_true_relative_improvement = -1.0;
void CED_SetProxyReduceTrueCheckHint(bool reduce_checks, double strength) {
  g_proxy_policy_reduce_checks = reduce_checks;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  if (!std::isfinite(strength))
    strength = 0.0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (strength < 0.0)
    strength = 0.0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  else if (strength > kProxyEnsembleSize)
    strength = kProxyEnsembleSize;
  g_proxy_policy_strength = strength;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
void CED_SetProxyDisagreementAuditHint(bool enabled) {
  g_proxy_policy_disagreement_audit = enabled;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
void CED_ClearProxyPolicyHint() {
  g_proxy_policy_reduce_checks = false;
  g_proxy_policy_strength = 0.0;
  g_proxy_policy_disagreement_audit = false;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
void CED_ForceNextProxyTrueEvaluation() {
  g_force_next_proxy_true_evaluation = true;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static bool ConsumeForcedProxyTrueEvaluation() {
  const bool forced = g_force_next_proxy_true_evaluation;
  g_force_next_proxy_true_evaluation = false;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return forced;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
int CED_ProxyTrueEvaluationCount() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return SharedAdaptiveProxyState().true_eval_count;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
int CED_ProxyExactCacheHitCount() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return SharedAdaptiveProxyState().exact_cache_hit_count;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
double CED_LastProxyTrueRelativeImprovement() {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return g_last_proxy_true_relative_improvement;
}

// 段落说明：计算、比较或保存候选解质量；最终报告值仍由真实调度目标复核。
template <typename Evaluate>
static double ReuseExactTrueValue(AdaptiveProxyState& state,
                                  const double* candidate, int variable_count,
                                  Evaluate evaluate) {
  // A pair of independent 64-bit fingerprints avoids retaining and repeatedly
  // scanning up to 24 complete high-dimensional solutions.  This preserves
  // duplicate-evaluation reuse while keeping the cache size independent of T.
  uint64_t hash_a = 1469598103934665603ULL;
  uint64_t hash_b = 0x9e3779b97f4a7c15ULL;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < variable_count; ++i) {
    uint64_t bits = 0;
    std::memcpy(&bits, candidate + i, sizeof(bits));
    hash_a ^= bits;
    hash_a *= 1099511628211ULL;
    bits += 0x9e3779b97f4a7c15ULL + (hash_b << 6) + (hash_b >> 2);
    hash_b ^= bits;
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (size_t i = 0; i < state.exact_true_hash_a.size(); i++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (state.exact_true_hash_a[i] == hash_a &&
        state.exact_true_hash_b[i] == hash_b) {
      state.exact_cache_hit_count++;
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return state.exact_true_values[i];
    }
  }
  const double value = evaluate();
  const int capacity = ProxyArchiveSize();
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if ((int)state.exact_true_hash_a.size() < capacity) {
    state.exact_true_hash_a.push_back(hash_a);
    state.exact_true_hash_b.push_back(hash_b);
    state.exact_true_values.push_back(value);
  } else if (capacity > 0) {
    const int replace = state.true_eval_count % capacity;
    state.exact_true_hash_a[replace] = hash_a;
    state.exact_true_hash_b[replace] = hash_b;
    state.exact_true_values[replace] = value;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return value;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static void EnsureAdaptiveProxyState(AdaptiveProxyState& state, CETask* task,
                                     int Cnum, int Enum, int Dnum,
                                     int CE_Tnum) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (state.last_task == task && state.last_cnum == Cnum &&
      state.last_enum == Enum && state.last_dnum == Dnum &&
      state.last_tnum == CE_Tnum)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  state.Reset();
  state.last_task = task;
  state.last_cnum = Cnum;
  state.last_enum = Enum;
  state.last_dnum = Dnum;
  state.last_tnum = CE_Tnum;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static inline double AdaptiveFallbackValue(const AdaptiveProxyState& state) {
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (state.best_true < 1e299)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return state.best_true * 1.05;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 1.0;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static inline void
EstimateLegacyProxyComponents(const AdaptiveProxyState& state,
                              const vector<double>& feature,
                              double (&component)[kProxyEnsembleSize]) {
  int nearest = -1;
  double nearest_dist = 1e300;
  double wsum = 0.0;
  double vsum = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (size_t i = 0; i < state.samples.size(); i++) {
    double dist = KrigingProxyDistance(feature, state.samples[i]);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (dist < nearest_dist) {
      nearest_dist = dist;
      nearest = (int)i;
    }
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (state.samples.size() >= 2) {
      double weight = exp(-dist / ProxyKernelBandwidth());
      wsum += weight;
      vsum += weight * state.values[i];
    }
  }
  double cluster = nearest < 0
                       ? AdaptiveFallbackValue(state)
                       : state.values[nearest] *
                             (1.0 + ProxyDistanceInflation() * nearest_dist);

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  double classifier = cluster;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (state.good_class.count >= 2 && state.bad_class.count >= 2) {
    double good_dist = KrigingProxyDistance(feature, state.good_class.center);
    double bad_dist = KrigingProxyDistance(feature, state.bad_class.center);
    classifier =
        bad_dist < good_dist ? state.bad_class.mean : state.good_class.mean;
  }

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  double rbf = cluster;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (state.samples.size() >= 2 && wsum > 1e-12)
    rbf = vsum / wsum;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  component[0] = cluster;
  component[1] = classifier;
  component[2] = rbf;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static inline double EstimateLegacyHybridProxy(const AdaptiveProxyState& state,
                                               const vector<double>& feature) {
  double component[kProxyEnsembleSize];
  EstimateLegacyProxyComponents(state, feature, component);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (state.ensemble_observations == 0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return (component[0] + component[1] + component[2]) / 3.0;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  const double learning_rate =
      sqrt(8.0 * log(static_cast<double>(kProxyEnsembleSize)) /
           static_cast<double>(state.ensemble_observations));
  double weighted_sum = 0.0;
  double weight_sum = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int model = 0; model < kProxyEnsembleSize; model++) {
    const double mean_loss =
        state.ensemble_loss[model] / state.ensemble_observations;
    const double weight = exp(-learning_rate * mean_loss);
    weighted_sum += weight * component[model];
    weight_sum += weight;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return weight_sum > 0.0 ? weighted_sum / weight_sum : component[0];
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double EstimateResidualGbdt(const AdaptiveProxyState& state,
                                   const vector<double>& feature) {
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (state.residual_stumps.empty())
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.0;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  double residual = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (size_t i = 0; i < state.residual_stumps.size(); i++) {
    const ProxyStump& stump = state.residual_stumps[i];
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (stump.feature < 0 || stump.feature >= (int)feature.size())
      continue;
    residual += feature[stump.feature] <= stump.threshold ? stump.left_value
                                                          : stump.right_value;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return std::isfinite(residual) ? residual : 0.0;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static inline double EstimateHybridProxy(const AdaptiveProxyState& state,
                                         const vector<double>& feature) {
  double legacy = EstimateLegacyHybridProxy(state, feature);
  double corrected = legacy + EstimateResidualGbdt(state, feature);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(corrected) || corrected <= 0.0)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return legacy;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return corrected;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static void RebuildResidualGbdt(AdaptiveProxyState& state) {
  const int sample_count = static_cast<int>(state.values.size());
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (state.residual_model_sample_count == sample_count &&
      state.residual_model_build_eval > 0 &&
      state.true_eval_count - state.residual_model_build_eval <
          sample_count / ProxyResidualStumpCount())
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  state.residual_stumps.clear();
  state.residual_model_sample_count = sample_count;
  state.residual_model_build_eval = state.true_eval_count;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  const int stump_count = ProxyResidualStumpCount();
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (sample_count < 2 * stump_count || state.samples.empty() ||
      state.samples[0].empty())
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  vector<double> residual = state.residual_values;
  const int feature_count = static_cast<int>(state.samples[0].size());
  const double shrinkage = 1.0 / std::sqrt((double)stump_count);

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  for (int tree = 0; tree < stump_count; tree++) {
    double best_sse = 1e300;
    ProxyStump best = {-1, 0.0, 0.0, 0.0};

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    for (int feature_index = 0; feature_index < feature_count;
         feature_index++) {
      vector<pair<double, int>> ordered(sample_count);
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int i = 0; i < sample_count; i++)
        ordered[i] = std::make_pair(state.samples[i][feature_index], i);
      std::sort(ordered.begin(), ordered.end());

      // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
      double total_sum = 0.0;
      double total_sq = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int i = 0; i < sample_count; i++) {
        total_sum += residual[i];
        total_sq += residual[i] * residual[i];
      }

      // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
      double left_sum = 0.0;
      double left_sq = 0.0;
      // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
      for (int split = 1; split < sample_count; split++) {
        const int sample_index = ordered[split - 1].second;
        left_sum += residual[sample_index];
        left_sq += residual[sample_index] * residual[sample_index];

        // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
        if (ordered[split - 1].first == ordered[split].first)
          continue;

        // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
        const int left_count = split;
        const int right_count = sample_count - split;
        const double right_sum = total_sum - left_sum;
        const double right_sq = total_sq - left_sq;
        const double left_mean = left_sum / left_count;
        const double right_mean = right_sum / right_count;
        const double left_sse = left_sq - left_sum * left_sum / left_count;
        const double right_sse = right_sq - right_sum * right_sum / right_count;
        const double sse = left_sse + right_sse;

        // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
        if (sse < best_sse) {
          best_sse = sse;
          best.feature = feature_index;
          best.threshold =
              0.5 * (ordered[split - 1].first + ordered[split].first);
          best.left_value = shrinkage * left_mean;
          best.right_value = shrinkage * right_mean;
        }
      }
    }

    // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
    if (best.feature < 0)
      break;

    // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
    state.residual_stumps.push_back(best);
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < sample_count; i++) {
      residual[i] -= state.samples[i][best.feature] <= best.threshold
                         ? best.left_value
                         : best.right_value;
    }
  }
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double ProxyRankKendallTau(const AdaptiveProxyState& state) {
  const int n = static_cast<int>(state.rank_proxy_values.size());
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (n < 2)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.0;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  int concordant = 0;
  int discordant = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < n; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = i + 1; j < n; j++) {
      double proxy_diff =
          state.rank_proxy_values[i] - state.rank_proxy_values[j];
      double true_diff = state.rank_true_values[i] - state.rank_true_values[j];
      double pair_product = proxy_diff * true_diff;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      if (pair_product > 0.0)
        concordant++;
      // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
      else if (pair_product < 0.0)
        discordant++;
    }
  }

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  const int pair_count = n * (n - 1) / 2;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return pair_count > 0 ? 2.0 * (concordant - discordant) /
                              static_cast<double>(n * (n - 1))
                        : 0.0;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static void RecordProxyRankCheck(AdaptiveProxyState& state, double proxy_value,
                                 double true_value) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(proxy_value) || !std::isfinite(true_value))
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  const int rank_window = ProxyRankWindow();
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if ((int)state.rank_proxy_values.size() < rank_window) {
    state.rank_proxy_values.push_back(proxy_value);
    state.rank_true_values.push_back(true_value);
  } else {
    int replace = state.true_eval_count % rank_window;
    state.rank_proxy_values[replace] = proxy_value;
    state.rank_true_values[replace] = true_value;
  }
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static inline bool NeedTrueProxyCheck(const AdaptiveProxyState& state,
                                      double proxy_value) {
  const int feasible_anchor_limit = ProxyFeasibleAnchorLimit();
  const int true_period = ProxyPeriodicCheckInterval();
  const bool periodic_check =
      true_period > 0 && state.eval_count % true_period == 0;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (state.last_tnum >= 100000 && g_proxy_policy_reduce_checks &&
      state.best_true < 1e299)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return periodic_check;
  bool verify =
      periodic_check ||
      (state.best_true >= 1e299 && state.eval_count <= feasible_anchor_limit);
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (state.best_true < 1e299) {
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (proxy_value < state.best_true * (1.0 - ProxyOptimismMargin()))
      verify = true;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (state.error > ProxyHighErrorBoundary() &&
        state.eval_count % ProxyRankWindow() == 0)
      verify = true;
  }
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (state.rank_proxy_values.size() >= kProxyBinsPerGroup) {
    double tau = ProxyRankKendallTau(state);
    // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
    if (tau < 0.0 && state.eval_count % (kProxyFeatureGroups / 2) == 0)
      verify = true;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    else if (tau > 0.0 && state.error < ProxyLowErrorBoundary() &&
             proxy_value >=
                 state.best_true * (1.0 + ProxyPoorPredictionMargin()))
      verify = false;
  }
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (state.best_true < 1e299 && g_proxy_policy_reduce_checks &&
      state.error < ProxyLowErrorBoundary()) {
    double conservative_band =
        state.best_true *
        (1.0 + ProxyConservativeMargin() * (1.0 + g_proxy_policy_strength));
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (proxy_value >= conservative_band)
      verify = false;
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return verify;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static bool PolicyRequestsDisagreementAudit(
    const AdaptiveProxyState& state, const vector<double>& feature,
    double proxy_value) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!g_proxy_policy_disagreement_audit || g_proxy_policy_reduce_checks ||
      state.best_true >= 1e299 ||
      !std::isfinite(proxy_value))
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return false;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  double component[kProxyEnsembleSize];
  EstimateLegacyProxyComponents(state, feature, component);
  double mean = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (double value : component)
    mean += value / kProxyEnsembleSize;
  double variance = 0.0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (double value : component) {
    const double difference = value - mean;
    variance += difference * difference / kProxyEnsembleSize;
  }
  const double disagreement =
      sqrt(max(0.0, variance)) / (fabs(mean) + 1e-12);
  const double learned_error = max(0.0, state.error);
  const bool uncertain = disagreement > learned_error;
  // Use the online mean relative proxy error as a calibrated uncertainty
  // radius.  Inter-model disagreement decides whether the prediction is
  // uncertain, while the archive-derived error controls how far beyond the
  // current verified best an exact audit may look.
  const bool potentially_better =
      proxy_value <= state.best_true * (1.0 + learned_error);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return uncertain && potentially_better;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static void LearnAdaptiveProxyValue(AdaptiveProxyState& state,
                                    const vector<double>& feature,
                                    double proxy_value, double true_value) {
  double component[kProxyEnsembleSize];
  EstimateLegacyProxyComponents(state, feature, component);
  const double loss_scale = fabs(true_value) + 1e-9;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int model = 0; model < kProxyEnsembleSize; model++) {
    const double relative_error = (component[model] - true_value) / loss_scale;
    state.ensemble_loss[model] += relative_error * relative_error;
  }
  state.ensemble_observations++;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  double scale = fabs(true_value) + 1e-9;
  double rel_error = fabs(true_value - proxy_value) / scale;
  const double observations =
      static_cast<double>(max(1, state.ensemble_observations));
  state.error += (rel_error - state.error) / observations;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (true_value < state.best_true)
    state.best_true = true_value;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  const int sample_cap = ProxyArchiveSize();
  double legacy_value = EstimateLegacyHybridProxy(state, feature);
  double residual_value =
      std::isfinite(legacy_value) ? true_value - legacy_value : 0.0;
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if ((int)state.samples.size() < sample_cap) {
    state.samples.push_back(feature);
    state.values.push_back(true_value);
    state.residual_values.push_back(residual_value);
  } else {
    int replace = state.eval_count % sample_cap;
    state.samples[replace] = feature;
    state.values[replace] = true_value;
    state.residual_values[replace] = residual_value;
  }

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  vector<double> ordered_values = state.values;
  nth_element(ordered_values.begin(),
              ordered_values.begin() + ordered_values.size() / 2,
              ordered_values.end());
  const double median_value = ordered_values[ordered_values.size() / 2];
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (true_value <= median_value || state.good_class.count == 0)
    UpdateClassifierProxyClass(state.good_class, feature, true_value);
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    UpdateClassifierProxyClass(state.bad_class, feature, true_value);
  RebuildResidualGbdt(state);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
static double RunSingleParallelProxy(
    AdaptiveProxyState& state, double* var, int Cnum, int Enum, int Dnum,
    int CE_Tnum, int M_Jnum, int M_OPTnum, CETask* CETask_Property,
    double* MTask_Time, DistanceValue** EtoD_Distance,
    DistanceValue** DtoD_Distance, vector<int>* AvailDeviceList,
    double* EnergyList, vector<int>* CloudDevices, vector<int>* EdgeDevices,
    vector<int>* CloudLoad, vector<int>* EdgeLoad, vector<int>* DeviceLoad,
    vector<int>* CETask_coDevice, map<int, double>* Edge_Device_comm,
    double** ST, double** ET, double* CE_ST, double* CE_ET) {
  EnsureAdaptiveProxyState(state, CETask_Property, Cnum, Enum, Dnum, CE_Tnum);
  g_last_proxy_true_relative_improvement = -1.0;
  const int variable_count = CE_Tnum * 2 + M_Jnum * M_OPTnum * 2;
  auto exact_true_value = [&]() {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ReuseExactTrueValue(state, var, variable_count, [&]() {
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return TrueScheduleValue(
          var, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
          MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
          CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
          CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
    });
  };

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  vector<double> feature;
  BuildKrigingProxyFeature(var, CE_Tnum, M_Jnum, M_OPTnum, feature);

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  state.eval_count++;
  const int feasible_anchor_limit = ProxyFeasibleAnchorLimit();
  const bool force_true =
      ConsumeForcedProxyTrueEvaluation() ||
      state.eval_count <= ProxyInitialDesignSize() ||
      (state.best_true >= 1e299 && state.eval_count <= feasible_anchor_limit);

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  bool proxy_is_true = force_true;
  double proxy_value =
      force_true
          ? (state.true_eval_count++, exact_true_value())
          : EstimateHybridProxy(state, feature);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(proxy_value))
    proxy_value = 1e300;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  bool verify = force_true || NeedTrueProxyCheck(state, proxy_value) ||
                PolicyRequestsDisagreementAudit(state, feature, proxy_value);
  double returned = proxy_value;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (verify) {
    double true_value = proxy_value;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!proxy_is_true) {
      state.true_eval_count++;
      true_value = exact_true_value();
      RecordProxyRankCheck(state, proxy_value, true_value);
    }
    const double previous_best_true = state.best_true;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (previous_best_true < 1e299 && true_value < previous_best_true)
      g_last_proxy_true_relative_improvement =
          (previous_best_true - true_value) /
          (fabs(previous_best_true) + 1e-12);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      g_last_proxy_true_relative_improvement = 0.0;
    LearnAdaptiveProxyValue(state, feature, proxy_value, true_value);
    returned = true_value;
  } else if (state.best_true < 1e299) {
    double guard = state.best_true * (1.0 + ProxyGuardScale() * state.error);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (proxy_value < guard)
      returned = guard;
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  return std::isfinite(returned) ? returned : 1e300;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
template <typename Materialize>
static double RunSingleParallelProxyWithFeature(
    AdaptiveProxyState& state, const vector<double>& feature,
    Materialize materialize, int Cnum, int Enum, int Dnum, int CE_Tnum,
    int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time,
    DistanceValue** EtoD_Distance, DistanceValue** DtoD_Distance,
    vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices,
    vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad,
    vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
    map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST,
    double* CE_ET) {
  state.eval_count++;
  g_last_proxy_true_relative_improvement = -1.0;
  const int variable_count = CE_Tnum * 2 + M_Jnum * M_OPTnum * 2;
  auto exact_true_value = [&]() {
    double* candidate = materialize();
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return ReuseExactTrueValue(state, candidate, variable_count, [&]() {
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return TrueScheduleValue(
          candidate, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
          CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
          AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad,
          EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET,
          CE_ST, CE_ET);
    });
  };
  const int feasible_anchor_limit = ProxyFeasibleAnchorLimit();
  const bool force_true =
      ConsumeForcedProxyTrueEvaluation() ||
      state.eval_count <= ProxyInitialDesignSize() ||
      (state.best_true >= 1e299 && state.eval_count <= feasible_anchor_limit);

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  bool proxy_is_true = force_true;
  double proxy_value =
      force_true
          ? (state.true_eval_count++, exact_true_value())
          : EstimateHybridProxy(state, feature);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!std::isfinite(proxy_value))
    proxy_value = 1e300;

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  bool verify = force_true || NeedTrueProxyCheck(state, proxy_value) ||
                PolicyRequestsDisagreementAudit(state, feature, proxy_value);
  double returned = proxy_value;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (verify) {
    double true_value = proxy_value;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (!proxy_is_true) {
      state.true_eval_count++;
      true_value = exact_true_value();
      RecordProxyRankCheck(state, proxy_value, true_value);
    }
    const double previous_best_true = state.best_true;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (previous_best_true < 1e299 && true_value < previous_best_true)
      g_last_proxy_true_relative_improvement =
          (previous_best_true - true_value) /
          (fabs(previous_best_true) + 1e-12);
    // 控制说明：条件不成立时执行互斥的备用处理路径。
    else
      g_last_proxy_true_relative_improvement = 0.0;
    LearnAdaptiveProxyValue(state, feature, proxy_value, true_value);
    returned = true_value;
  } else if (state.best_true < 1e299) {
    double guard = state.best_true * (1.0 + ProxyGuardScale() * state.error);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (proxy_value < guard)
      returned = guard;
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  return std::isfinite(returned) ? returned : 1e300;
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
double CED_Schedule_ParallelProxy(
    double* var, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum,
    int M_OPTnum, CETask* CETask_Property, double* MTask_Time,
    DistanceValue** EtoD_Distance, DistanceValue** DtoD_Distance,
    vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices,
    vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad,
    vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
    map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST,
    double* CE_ET) {
  AdaptiveProxyState& state = SharedAdaptiveProxyState();
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return RunSingleParallelProxy(
      state, var, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum, CETask_Property,
      MTask_Time, EtoD_Distance, DtoD_Distance, AvailDeviceList, EnergyList,
      CloudDevices, EdgeDevices, CloudLoad, EdgeLoad, DeviceLoad,
      CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST, CE_ET);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
double CED_Schedule_ParallelProxy_DenseBlend(
    double* base_var, double* target_var, double alpha, double* materialized,
    int* materialized_ready, int Cnum, int Enum, int Dnum, int CE_Tnum,
    int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time,
    DistanceValue** EtoD_Distance, DistanceValue** DtoD_Distance,
    vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices,
    vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad,
    vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
    map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST,
    double* CE_ET) {
  AdaptiveProxyState& state = SharedAdaptiveProxyState();
  EnsureAdaptiveProxyState(state, CETask_Property, Cnum, Enum, Dnum, CE_Tnum);

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  auto value_at = [&](int index) {
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return UnitValue(base_var[index] +
                     alpha * (target_var[index] - base_var[index]));
  };

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  vector<double> feature;
  BuildKrigingProxyFeatureFromValue(CE_Tnum, M_Jnum, M_OPTnum, feature,
                                    value_at);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const int nvar = CE_Tnum * 2 + M_Jnum * M_OPTnum * 2;
  auto materialize = [&]() {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (materialized != nullptr && materialized_ready != nullptr &&
        *materialized_ready)
      return materialized;
    static vector<double> fallback_materialized;
    double* target = materialized;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (target == nullptr) {
      fallback_materialized.resize(nvar);
      target = fallback_materialized.data();
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < nvar; i++)
      target[i] = value_at(i);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (materialized != nullptr && materialized_ready != nullptr)
      *materialized_ready = 1;
    return target;
  };

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  return RunSingleParallelProxyWithFeature(
      state, feature, materialize, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
      CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
      AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad,
      EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST,
      CE_ET);
}

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
double CED_Schedule_ParallelProxy_DenseOpposition(
    double* base_var, double* gbest_var, double* ibest_var,
    double opposition_alpha, double beta, int opposition, double* materialized,
    int* materialized_ready, int Cnum, int Enum, int Dnum, int CE_Tnum,
    int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time,
    DistanceValue** EtoD_Distance, DistanceValue** DtoD_Distance,
    vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices,
    vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad,
    vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
    map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST,
    double* CE_ET) {
  AdaptiveProxyState& state = SharedAdaptiveProxyState();
  EnsureAdaptiveProxyState(state, CETask_Property, Cnum, Enum, Dnum, CE_Tnum);

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  auto value_at = [&](int index) {
    double opposite = 1.0 - base_var[index];
    double elite = opposition_alpha * gbest_var[index] +
                   (1.0 - opposition_alpha) * ibest_var[index];
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (opposition)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return UnitValue(0.5 * opposite + 0.5 * elite);
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return UnitValue(base_var[index] + beta * (elite - opposite));
  };

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  vector<double> feature;
  BuildKrigingProxyFeatureFromValue(CE_Tnum, M_Jnum, M_OPTnum, feature,
                                    value_at);

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  const int nvar = CE_Tnum * 2 + M_Jnum * M_OPTnum * 2;
  auto materialize = [&]() {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (materialized != nullptr && materialized_ready != nullptr &&
        *materialized_ready)
      return materialized;
    static vector<double> fallback_materialized;
    double* target = materialized;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (target == nullptr) {
      fallback_materialized.resize(nvar);
      target = fallback_materialized.data();
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < nvar; i++)
      target[i] = value_at(i);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (materialized != nullptr && materialized_ready != nullptr)
      *materialized_ready = 1;
    return target;
  };

  // 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
  return RunSingleParallelProxyWithFeature(
      state, feature, materialize, Cnum, Enum, Dnum, CE_Tnum, M_Jnum, M_OPTnum,
      CETask_Property, MTask_Time, EtoD_Distance, DtoD_Distance,
      AvailDeviceList, EnergyList, CloudDevices, EdgeDevices, CloudLoad,
      EdgeLoad, DeviceLoad, CETask_coDevice, Edge_Device_comm, ST, ET, CE_ST,
      CE_ET);
}
