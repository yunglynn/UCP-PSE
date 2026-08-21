// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef CED_SCHEDULE_CONFIG_H
#define CED_SCHEDULE_CONFIG_H

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef TNUM
#define TNUM 10000
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef CNUM
#if TNUM == 1000
#define CNUM 16
#elif TNUM == 10000
#define CNUM 160
#elif TNUM == 100000
#define CNUM 1600
#elif TNUM == 1000000
#define CNUM 16000
#else
#error "Unsupported TNUM for the regional industrial benchmark"
#endif
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef ENUM
#if TNUM == 1000
#define ENUM 4
#elif TNUM == 10000
#define ENUM 40
#elif TNUM == 100000
#define ENUM 400
#elif TNUM == 1000000
#define ENUM 4000
#else
#error "Unsupported TNUM for the regional industrial benchmark"
#endif
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef DNUM
#if TNUM == 1000
#define DNUM 64
#elif TNUM == 10000
#define DNUM 640
#elif TNUM == 100000
#define DNUM 6400
#elif TNUM == 1000000
#define DNUM 64000
#else
#error "Unsupported TNUM for the regional industrial benchmark"
#endif
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef MOPT_NUM
#define MOPT_NUM 5
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef MAXGEN
#define MAXGEN 50
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef POPSIZE
#define POPSIZE 8
#endif

// 段落说明：定义或选择编译期配置分支；未显式覆盖时保持论文/Runbook 默认值。
#ifndef VERBOSE_OUTPUT
#define VERBOSE_OUTPUT 0
#endif

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
#endif
