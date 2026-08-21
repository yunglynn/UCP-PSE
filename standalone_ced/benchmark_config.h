#ifndef STANDALONE_CED_BENCHMARK_CONFIG_H
#define STANDALONE_CED_BENCHMARK_CONFIG_H

#ifndef TNUM
#define TNUM 1000
#endif
#ifndef MAXGEN
#define MAXGEN 50
#endif
#ifndef MOPT_NUM
#define MOPT_NUM 5
#endif

#include <cstdlib>

namespace standalone_ced {

inline constexpr int kPopulationSize = 8;
inline unsigned experiment_seed() {
  const char* value = std::getenv("CED_SEED");
  return value == nullptr ? 20260616U
                          : static_cast<unsigned>(std::strtoul(value, nullptr, 10));
}
inline constexpr const char* kPowerPath =
    "/Users/lailiyuanjun/Desktop/CED_schedule/CED_schedule/Power_Consumption.txt";

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
#else
inline constexpr int kCloudCount = 16;
inline constexpr int kEdgeCount = 4;
inline constexpr int kDeviceCount = 64;
inline constexpr const char* kDataPath =
    "/Users/lailiyuanjun/Desktop/data_generator/datamatrix_1000";
#endif

} // namespace standalone_ced

#endif
