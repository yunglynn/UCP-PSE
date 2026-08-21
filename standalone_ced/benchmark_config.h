// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef STANDALONE_CED_BENCHMARK_CONFIG_H
#define STANDALONE_CED_BENCHMARK_CONFIG_H

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef TNUM
#define TNUM 1000
#endif
#ifndef MAXGEN
#define MAXGEN 50
#endif
#ifndef MOPT_NUM
#define MOPT_NUM 5
#endif

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <cstdlib>

// 段落说明：进入局部命名空间，避免辅助符号污染公共接口。
namespace standalone_ced {

// 段落说明：读取可选环境覆盖；正式基线运行前必须按 Runbook 清空未授权覆盖。
inline constexpr int kPopulationSize = 8;
inline unsigned experiment_seed() {
  const char* value = std::getenv("CED_SEED");
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return value == nullptr ? 20260616U
                          : static_cast<unsigned>(std::strtoul(value, nullptr, 10));
}
inline constexpr const char* kPowerPath =
    "/Users/lailiyuanjun/Desktop/CED_schedule/CED_schedule/Power_Consumption.txt";

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#if TNUM >= 1000000
inline constexpr int kCloudCount = 16000;
inline constexpr int kEdgeCount = 4000;
inline constexpr int kDeviceCount = 64000;
inline constexpr const char* kDataPath =
    "/Users/lailiyuanjun/Desktop/data_generator/datamatrix_1000000_compact";
#elif TNUM >= 100000
inline constexpr int kCloudCount = 1600;
inline constexpr int kEdgeCount = 400;
inline constexpr int kDeviceCount = 6400;
inline constexpr const char* kDataPath =
    "/Users/lailiyuanjun/Desktop/data_generator/datamatrix_100000";
#elif TNUM >= 10000
inline constexpr int kCloudCount = 160;
inline constexpr int kEdgeCount = 40;
inline constexpr int kDeviceCount = 640;
inline constexpr const char* kDataPath =
    "/Users/lailiyuanjun/Desktop/data_generator/datamatrix_10000";
// 控制说明：选择当前编译配置对应的实现路径。
#else
inline constexpr int kCloudCount = 16;
inline constexpr int kEdgeCount = 4;
inline constexpr int kDeviceCount = 64;
inline constexpr const char* kDataPath =
    "/Users/lailiyuanjun/Desktop/data_generator/datamatrix_1000";
#endif

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
} // namespace standalone_ced

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
#endif
