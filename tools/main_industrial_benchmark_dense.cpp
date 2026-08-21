// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
using namespace std;

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace {
constexpr int kTasksPerFactory = 1000;
constexpr int kOperationsPerJob = 5;
constexpr int kCloudNodesPerFactory = 16;
constexpr int kEdgeNodesPerFactory = 4;
constexpr int kDevicesPerFactory = 64;
constexpr double kFactoryLengthM = 180.0;
constexpr double kFactoryWidthM = 80.0;
constexpr double kFactoryPitchM = 2000.0;
constexpr unsigned kSeed = 20260807U;

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct Point {
  double x;
  double y;
};

// 段落说明：实现 `distance`：完成该函数负责的数据准备、算法步骤和状态返回。
double distance(const Point& a, const Point& b) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return hypot(a.x - b.x, a.y - b.y);
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
template <typename T>
void write_vector(ofstream& out, const vector<T>& values) {
  out << values.size() << '\n';
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (const T& value : values)
    out << value << "    ";
  out << "\n\n";
}

// 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
vector<int> choose_operation_devices(int factory, int job_local, int operation) {
  vector<int> candidates;
  const int device_base = factory * kDevicesPerFactory;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int d = 0; d < kDevicesPerFactory; ++d) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (d % kOperationsPerJob == operation)
      candidates.push_back(device_base + d);
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  vector<int> selected;
  const int start = (job_local * 7 + operation * 11) % candidates.size();
  const int stride = 7;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int k = 0; k < 3; ++k)
    selected.push_back(candidates[(start + k * stride) % candidates.size()]);
  sort(selected.begin(), selected.end());
  selected.erase(unique(selected.begin(), selected.end()), selected.end());
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (selected.size() != 3)
    throw runtime_error("Failed to select three distinct manufacturing devices");
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return selected;
}

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
void write_metadata(const string& output, int tasks, int factories, int clouds,
                    int edges, int devices) {
  ofstream meta(output + ".meta");
  meta << "benchmark=regional_multi_factory_iiot\n"
       << "seed=" << kSeed << '\n'
       << "tasks=" << tasks << '\n'
       << "manufacturing_jobs=" << tasks << '\n'
       << "operations_per_job=" << kOperationsPerJob << '\n'
       << "factories=" << factories << '\n'
       << "cloud_compute_nodes=" << clouds << '\n'
       << "edge_compute_nodes=" << edges << '\n'
       << "industrial_devices=" << devices << '\n'
       << "factory_footprint_m=" << kFactoryLengthM << "x"
       << kFactoryWidthM << '\n'
       << "factory_pitch_m=" << kFactoryPitchM << '\n'
       << "edge_nodes_per_factory=" << kEdgeNodesPerFactory << '\n'
       << "devices_per_factory=" << kDevicesPerFactory << '\n'
       << "eligible_devices_per_operation=3\n"
       << "source_note=Factory footprint and device-scale interpretation follow the "
          "5G-ACIA industrial traffic-model example; topology and workload ratios "
          "are explicit synthetic benchmark assumptions.\n";
}
} // namespace

// 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
int main(int argc, char** argv) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " TNUM OUTPUT_PATH\n";
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1;
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  const int task_count = stoi(argv[1]);
  // 控制说明：检查容器规模或样本数量，避免越界和无效统计。
  if (task_count <= 0 || task_count % kTasksPerFactory != 0) {
    cerr << "TNUM must be a positive multiple of " << kTasksPerFactory << "\n";
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1;
  }

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  const string output_path = argv[2];
  const int factory_count = task_count / kTasksPerFactory;
  const int cloud_count = factory_count * kCloudNodesPerFactory;
  const int edge_count = factory_count * kEdgeNodesPerFactory;
  const int device_count = factory_count * kDevicesPerFactory;
  const int factory_columns = static_cast<int>(ceil(sqrt(factory_count)));

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  vector<Point> factory_centers(factory_count);
  vector<Point> edge_locations(edge_count);
  vector<Point> device_locations(device_count);
  mt19937 rng(kSeed + static_cast<unsigned>(task_count));
  uniform_real_distribution<double> device_x(-kFactoryLengthM / 2.0,
                                              kFactoryLengthM / 2.0);
  uniform_real_distribution<double> device_y(-kFactoryWidthM / 2.0,
                                              kFactoryWidthM / 2.0);

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  const Point edge_offsets[kEdgeNodesPerFactory] = {
      {-45.0, -20.0}, {45.0, -20.0}, {-45.0, 20.0}, {45.0, 20.0}};
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int f = 0; f < factory_count; ++f) {
    const int row = f / factory_columns;
    const int col = f % factory_columns;
    factory_centers[f] = {col * kFactoryPitchM, row * kFactoryPitchM};
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int e = 0; e < kEdgeNodesPerFactory; ++e) {
      edge_locations[f * kEdgeNodesPerFactory + e] = {
          factory_centers[f].x + edge_offsets[e].x,
          factory_centers[f].y + edge_offsets[e].y};
    }
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int d = 0; d < kDevicesPerFactory; ++d) {
      device_locations[f * kDevicesPerFactory + d] = {
          factory_centers[f].x + device_x(rng),
          factory_centers[f].y + device_y(rng)};
    }
  }

  // 段落说明：处理当前边界条件或配置分支，避免无效索引、数值或不支持路径。
  ofstream out(output_path);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (!out) {
    cerr << "Cannot open output file: " << output_path << '\n';
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 1;
  }
  out << fixed << setprecision(3);

  // Edge-to-device distances.
  for (const Point& edge : edge_locations) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (const Point& device : device_locations)
      out << distance(edge, device) << "    ";
    out << '\n';
  }
  out << "\n\n";

  // Device-to-device distances.
  for (const Point& source : device_locations) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (const Point& target : device_locations)
      out << distance(source, target) << "    ";
    out << '\n';
  }
  out << "\n\n";

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  uniform_int_distribution<int> operation_time(30, 300);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < task_count * kOperationsPerJob; ++i)
    out << operation_time(rng) << "    ";
  out << "\n\n";

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  uniform_int_distribution<int> computation(50, 199);
  uniform_int_distribution<int> communication(500, 4999);
  uniform_real_distribution<double> probability(0.0, 1.0);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < task_count; ++i) {
    const int local = i % kTasksPerFactory;
    const int factory_base = (i / kTasksPerFactory) * kTasksPerFactory;
    vector<int> precedence;
    vector<int> interaction;
    vector<int> start_pre;
    vector<int> end_pre;

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    if (local > 0)
      precedence.push_back(i - 1);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (local > 2 && probability(rng) < 0.35)
      precedence.push_back(max(factory_base, i - 2 - static_cast<int>(rng() % 9)));
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (local > 0 && probability(rng) < 0.30)
      interaction.push_back(max(factory_base, i - 1 - static_cast<int>(rng() % 10)));
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (local > 0 && probability(rng) < 0.20)
      start_pre.push_back(max(factory_base, i - 1 - static_cast<int>(rng() % 10)));
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (local > 0 && probability(rng) < 0.20)
      end_pre.push_back(max(factory_base, i - 1 - static_cast<int>(rng() % 10)));

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    out << computation(rng) << "\n\n";
    out << communication(rng) << "\n\n";
    write_vector(out, precedence);
    write_vector(out, interaction);
    write_vector(out, start_pre);
    write_vector(out, end_pre);

    // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
    const double coupling_draw = probability(rng);
    const int job_constraint = coupling_draw < 0.30 ? 0
                               : coupling_draw < 0.55 ? 1
                               : coupling_draw < 0.80 ? 2
                                                      : 3;
    out << job_constraint << "\n\n";
  }

  // Three capability-compatible industrial devices per operation.
  for (int job = 0; job < task_count; ++job) {
    const int factory = job / kTasksPerFactory;
    const int job_local = job % kTasksPerFactory;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int operation = 0; operation < kOperationsPerJob; ++operation)
      write_vector(out, choose_operation_devices(factory, job_local, operation));
  }

  // Four on-premises edge nodes are available to every task in its factory.
  for (int task = 0; task < task_count; ++task) {
    const int factory = task / kTasksPerFactory;
    vector<int> available_edges;
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int e = 0; e < kEdgeNodesPerFactory; ++e)
      available_edges.push_back(factory * kEdgeNodesPerFactory + e);
    write_vector(out, available_edges);
  }
  out.close();

  // 段落说明：输出可审计的运行信息、指标或错误原因。
  write_metadata(output_path, task_count, factory_count, cloud_count,
                 edge_count, device_count);
  cout << "Generated " << output_path << ": tasks=" << task_count
       << ", factories=" << factory_count << ", cloud=" << cloud_count
       << ", edge=" << edge_count << ", devices=" << device_count << '\n';
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return 0;
}
