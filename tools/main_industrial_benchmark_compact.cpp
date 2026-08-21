/* Deterministic four-scale CED_COMPACT_V1 generator: validate scale, derive
 * resources, generate geometry/workloads/couplings, and write a .meta audit. */
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

using namespace std;

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

struct Point {
  double x;
  double y;
};

double distance(const Point& a, const Point& b) {
  return hypot(a.x - b.x, a.y - b.y);
}

template <typename T>
void write_vector(ofstream& out, const vector<T>& values) {
  out << values.size() << '\n';
  for (const T& value : values)
    out << value << "    ";
  out << "\n\n";
}

vector<int> choose_operation_devices(int factory, int job_local, int operation) {
  vector<int> candidates;
  const int device_base = factory * kDevicesPerFactory;
  for (int d = 0; d < kDevicesPerFactory; ++d) {
    if (d % kOperationsPerJob == operation)
      candidates.push_back(device_base + d);
  }

  vector<int> selected;
  const int start = (job_local * 7 + operation * 11) % candidates.size();
  const int stride = 7;
  for (int k = 0; k < 3; ++k)
    selected.push_back(candidates[(start + k * stride) % candidates.size()]);
  sort(selected.begin(), selected.end());
  selected.erase(unique(selected.begin(), selected.end()), selected.end());
  if (selected.size() != 3)
    throw runtime_error("Failed to select three distinct manufacturing devices");
  return selected;
}

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

int main(int argc, char** argv) {
  if (argc != 3) {
    cerr << "Usage: " << argv[0] << " TNUM OUTPUT_PATH\n";
    return 1;
  }

  const int task_count = stoi(argv[1]);
  if (task_count <= 0 || task_count % kTasksPerFactory != 0) {
    cerr << "TNUM must be a positive multiple of " << kTasksPerFactory << "\n";
    return 1;
  }

  const string output_path = argv[2];
  const int factory_count = task_count / kTasksPerFactory;
  const int cloud_count = factory_count * kCloudNodesPerFactory;
  const int edge_count = factory_count * kEdgeNodesPerFactory;
  const int device_count = factory_count * kDevicesPerFactory;
  const int factory_columns = static_cast<int>(ceil(sqrt(factory_count)));

  vector<Point> factory_centers(factory_count);
  vector<Point> edge_locations(edge_count);
  vector<Point> device_locations(device_count);
  mt19937 rng(kSeed + static_cast<unsigned>(task_count));
  uniform_real_distribution<double> device_x(-kFactoryLengthM / 2.0,
                                              kFactoryLengthM / 2.0);
  uniform_real_distribution<double> device_y(-kFactoryWidthM / 2.0,
                                              kFactoryWidthM / 2.0);

  const Point edge_offsets[kEdgeNodesPerFactory] = {
      {-45.0, -20.0}, {45.0, -20.0}, {-45.0, 20.0}, {45.0, 20.0}};
  for (int f = 0; f < factory_count; ++f) {
    const int row = f / factory_columns;
    const int col = f % factory_columns;
    factory_centers[f] = {col * kFactoryPitchM, row * kFactoryPitchM};
    for (int e = 0; e < kEdgeNodesPerFactory; ++e) {
      edge_locations[f * kEdgeNodesPerFactory + e] = {
          factory_centers[f].x + edge_offsets[e].x,
          factory_centers[f].y + edge_offsets[e].y};
    }
    for (int d = 0; d < kDevicesPerFactory; ++d) {
      device_locations[f * kDevicesPerFactory + d] = {
          factory_centers[f].x + device_x(rng),
          factory_centers[f].y + device_y(rng)};
    }
  }

  ofstream out(output_path);
  if (!out) {
    cerr << "Cannot open output file: " << output_path << '\n';
    return 1;
  }
  out << fixed << setprecision(3);

  // Compact geometry header. Distances are reconstructed on demand.
  out << "CED_COMPACT_V1\n";
  out << edge_count << ' ' << device_count << '\n';
  for (const Point& edge : edge_locations)
    out << edge.x << ' ' << edge.y << '\n';
  for (const Point& device : device_locations)
    out << device.x << ' ' << device.y << '\n';
  out << '\n';

  uniform_int_distribution<int> operation_time(30, 300);
  for (int i = 0; i < task_count * kOperationsPerJob; ++i)
    out << operation_time(rng) << "    ";
  out << "\n\n";

  uniform_int_distribution<int> computation(50, 199);
  uniform_int_distribution<int> communication(500, 4999);
  uniform_real_distribution<double> probability(0.0, 1.0);
  for (int i = 0; i < task_count; ++i) {
    const int local = i % kTasksPerFactory;
    const int factory_base = (i / kTasksPerFactory) * kTasksPerFactory;
    vector<int> precedence;
    vector<int> interaction;
    vector<int> start_pre;
    vector<int> end_pre;

    if (local > 0)
      precedence.push_back(i - 1);
    if (local > 2 && probability(rng) < 0.35)
      precedence.push_back(max(factory_base, i - 2 - static_cast<int>(rng() % 9)));
    if (local > 0 && probability(rng) < 0.30)
      interaction.push_back(max(factory_base, i - 1 - static_cast<int>(rng() % 10)));
    if (local > 0 && probability(rng) < 0.20)
      start_pre.push_back(max(factory_base, i - 1 - static_cast<int>(rng() % 10)));
    if (local > 0 && probability(rng) < 0.20)
      end_pre.push_back(max(factory_base, i - 1 - static_cast<int>(rng() % 10)));

    out << computation(rng) << "\n\n";
    out << communication(rng) << "\n\n";
    write_vector(out, precedence);
    write_vector(out, interaction);
    write_vector(out, start_pre);
    write_vector(out, end_pre);

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
    for (int operation = 0; operation < kOperationsPerJob; ++operation)
      write_vector(out, choose_operation_devices(factory, job_local, operation));
  }

  // Four on-premises edge nodes are available to every task in its factory.
  for (int task = 0; task < task_count; ++task) {
    const int factory = task / kTasksPerFactory;
    vector<int> available_edges;
    for (int e = 0; e < kEdgeNodesPerFactory; ++e)
      available_edges.push_back(factory * kEdgeNodesPerFactory + e);
    write_vector(out, available_edges);
  }
  out.close();

  write_metadata(output_path, task_count, factory_count, cloud_count,
                 edge_count, device_count);
  cout << "Generated " << output_path << ": tasks=" << task_count
       << ", factories=" << factory_count << ", cloud=" << cloud_count
       << ", edge=" << edge_count << ", devices=" << device_count << '\n';
  return 0;
}
