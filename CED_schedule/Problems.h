// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef CED_SCHEDULE_PROBLEMS_H
#define CED_SCHEDULE_PROBLEMS_H

// 段落说明：引入本段实现依赖的项目接口或 C++ 标准库组件。
#include <map>
#include <vector>
using namespace std;

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
typedef unsigned short DistanceValue;
// 控制说明：选择当前编译配置对应的实现路径。
#ifndef DISTANCE_SCALE
#define DISTANCE_SCALE 3.0
#endif
inline constexpr double kTransmitPowerDbm = 23.0;

// 段落说明：实现 `DecodeDistance`：完成该函数负责的数据准备、算法步骤和状态返回。
inline double DecodeDistance(DistanceValue value) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return static_cast<double>(value) * DISTANCE_SCALE;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
typedef struct {
  double Computation;
  double Communication;
  vector<int> Precedence;
  vector<int> Interact;
  vector<int> Start_Pre;
  vector<int> End_Pre;
  int Job_Constraints;
  vector<int> AvailEdgeServerList;
} CETask;

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct CEDLoadDistribution {
  double mean = 0.0;
  double standard_deviation = 0.0;
  double minimum = 0.0;
  double median = 0.0;
  double percentile95 = 0.0;
  double maximum = 0.0;
  double jain_index = 1.0;
  double total_count = 0.0;
  double used_count = 0.0;
  double active_jain_index = 0.0;
};

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
struct CEDDetailedMetrics {
  double objective = 0.0;
  double energy = 0.0;
  double makespan = 0.0;
  double average_operation_queue_wait = 0.0;
  double average_task_communication_time = 0.0;
  double total_transportation_time = 0.0;
  CEDLoadDistribution cloud_load;
  CEDLoadDistribution edge_load;
  CEDLoadDistribution device_load;
  CEDLoadDistribution factory_busy_time;
  CEDLoadDistribution factory_idle_time;
  CEDLoadDistribution all_resource_busy_time;
  CEDLoadDistribution all_resource_idle_time;
  CEDLoadDistribution cloud_busy_time;
  CEDLoadDistribution cloud_idle_time;
  CEDLoadDistribution edge_busy_time;
  CEDLoadDistribution edge_idle_time;
  CEDLoadDistribution device_busy_time;
  CEDLoadDistribution device_idle_time;
};

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
const CEDDetailedMetrics& CED_LastDetailedMetrics();

// 段落说明：维护轻量代理的特征、预测、误差或审计状态，用于减少昂贵的真实评估。
double CED_Schedule(double* var, int Cnum, int Enum, int Dnum, int CE_Tnum,
                    int M_Jnum, int M_OPTnum, CETask* CETask_Property,
                    double* MTask_Time, DistanceValue** EtoD_Distance,
                    DistanceValue** DtoD_Distance, vector<int>* AvailDeviceList,
                    double* EnergyList, vector<int>* CloudDevices,
                    vector<int>* EdgeDevices, vector<int>* CloudLoad,
                    vector<int>* EdgeLoad, vector<int>* DeviceLoad,
                    vector<int>* CETask_coDevice,
                    map<int, double>* Edge_Device_comm, double** ST,
                    double** ET, double* CE_ST, double* CE_ET);
double CED_Schedule_ParallelProxy(
    double* var, int Cnum, int Enum, int Dnum, int CE_Tnum, int M_Jnum,
    int M_OPTnum, CETask* CETask_Property, double* MTask_Time,
    DistanceValue** EtoD_Distance, DistanceValue** DtoD_Distance,
    vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices,
    vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad,
    vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
    map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST,
    double* CE_ET);
double CED_Schedule_ParallelProxy_DenseBlend(
    double* base_var, double* target_var, double alpha, double* materialized,
    int* materialized_ready, int Cnum, int Enum, int Dnum, int CE_Tnum,
    int M_Jnum, int M_OPTnum, CETask* CETask_Property, double* MTask_Time,
    DistanceValue** EtoD_Distance, DistanceValue** DtoD_Distance,
    vector<int>* AvailDeviceList, double* EnergyList, vector<int>* CloudDevices,
    vector<int>* EdgeDevices, vector<int>* CloudLoad, vector<int>* EdgeLoad,
    vector<int>* DeviceLoad, vector<int>* CETask_coDevice,
    map<int, double>* Edge_Device_comm, double** ST, double** ET, double* CE_ST,
    double* CE_ET);
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
    double* CE_ET);
void CED_SetProxyReduceTrueCheckHint(bool reduce_checks, double strength);
void CED_SetProxyDisagreementAuditHint(bool enabled);
void CED_ClearProxyPolicyHint();
void CED_ForceNextProxyTrueEvaluation();
int CED_ProxyTrueEvaluationCount();
double CED_ProxyBestTrueValue();
int CED_ProxyExactCacheHitCount();
double CED_LastProxyTrueRelativeImprovement();
void CED_SetCompactGeometry(const double* edge_x, const double* edge_y,
                            const double* device_x, const double* device_y,
                            int edge_count, int device_count);

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
#endif
