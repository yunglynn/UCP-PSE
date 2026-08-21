#ifndef CED_SCHEDULE_CONFIG_H
#define CED_SCHEDULE_CONFIG_H

#ifndef TNUM
#define TNUM 10000
#endif

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

#ifndef MOPT_NUM
#define MOPT_NUM 5
#endif

#ifndef MAXGEN
#define MAXGEN 50
#endif

#ifndef POPSIZE
#define POPSIZE 8
#endif

#ifndef VERBOSE_OUTPUT
#define VERBOSE_OUTPUT 0
#endif

#endif
