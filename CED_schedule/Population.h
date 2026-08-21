// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
#pragma once
#ifndef CED_SCHEDULE_POPULATION_H
#define CED_SCHEDULE_POPULATION_H
#include <cstring>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
using namespace std;
inline constexpr double kPi = 3.1415926;
inline constexpr int kArchiveSize = 10;

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> struct IndividualSnapshot {
  vector<T> pop;
  vector<T> newpop;
  double pop_fit;
  double newpop_fit;
};

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> class Population {
public:
  Population(int psize, int nn, T lb, T ub);
  virtual ~Population();

// 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
public:
  T** pop;
  double* pop_fit;
  T** newpop;
  double* newpop_fit;
  T* gbest;
  double gbest_fit;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  double** delta_pop;
  double* delta_fit;
  int delta_update_count;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  int Popsize;
  int Nvar;
  T Ubound;
  T Lbound;
  int cur_best;
  int cur_worst;
  T CRold, CRnew;

  // CMAES
  bool stored;
  double hold;

  // 段落说明：声明并初始化本阶段所需的参数、缓存或统计状态。
  long int t;
  long unsigned seed;
  long int aktseed;
  long int aktrand;
  long int rgrand[32];

// 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
public:
  T** CreateMatrix(int nRow, int nCol); // 多维数组创建
  void DeleteMatrix(T** ppT, int nRow); // 多维数组删除
  inline double randval(double low, double high);
  void heap_adjust(T** num, int s, int len, int cbit);
  void heap_sort(T** num, int len, int cbit);
  void swap(T&, T&);
  void worst_and_best(); // 种群最优及最差解
  void Elist();
  inline double average_fit();
  double pop_random_variance(int p_start, int p_end);
  double pop_random_entropy(int subn, int p_start, int p_end);
  void CRfit();
  IndividualSnapshot<T> SnapshotIndividual(int index);
  void RestorePopIndividual(int index, const IndividualSnapshot<T>& snapshot);

  // CMAES
  // double maxElement(const double* rgd, int len);
  // double minElement(const double* rgd, int len);
  double square(double d);
  double gauss(void);
  double uniform(void);
  double myhypot(double a, double b);
  double Euclidean_dis(double* a, double* b, int len);
};

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T>
Population<T>::Population(int psize, int nn, T lb, T ub)
    : pop(0), pop_fit(0), newpop(0), newpop_fit(0), gbest(0), gbest_fit(0),
      delta_pop(0), delta_fit(0), delta_update_count(0), Popsize(psize),
      Nvar(nn), Ubound(ub), Lbound(lb), cur_best(0), cur_worst(0), CRold(0),
      CRnew(0), stored(false), hold(0.0), t(0), seed(0), aktseed(0), aktrand(0),
      rgrand() {
  pop = CreateMatrix(Popsize, Nvar);
  newpop = CreateMatrix(Popsize, Nvar);
  gbest = new T[Nvar];
  pop_fit = new double[Popsize];
  newpop_fit = new double[Popsize];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; j++) {
      pop[i][j] = (T)randval(Lbound, Ubound);
    }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < Popsize; i++) {
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int j = 0; j < Nvar; j++) {
      newpop[i][j] = (T)randval(Lbound, Ubound);
    }
  }
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Nvar; i++) {
    gbest[i] = (T)randval(Lbound, Ubound);
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  delta_pop = nullptr;
  delta_fit = nullptr;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (Nvar < 10000000) {
    delta_pop = new double*[kArchiveSize];
    delta_fit = new double[kArchiveSize];
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < kArchiveSize; i++)
      delta_pop[i] = new double[Nvar * 2];
    // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
    for (int i = 0; i < kArchiveSize; i++)
      delta_fit[i] = 0;
  }
  delta_update_count = 0;

  // CMAES

  // 段落说明：按确定性种子驱动随机采样，使同一配置和种子能够重复。
  t = rand() + 1;
  seed = (long unsigned)(t < 0 ? -t : t);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (seed < 1)
    seed = 1;
  aktseed = seed;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 39; i >= 0; --i) {
    long tmp = aktseed / 127773;
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (aktseed < 0)
      aktseed += 2147483647;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (i < 32)
      rgrand[i] = aktseed;
  }
  aktrand = rgrand[0];
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> Population<T>::~Population() {
  DeleteMatrix(pop, Popsize);
  DeleteMatrix(newpop, Popsize);
  delete[] gbest;
  delete[] pop_fit;
  delete[] newpop_fit;
  DeleteMatrix(delta_pop, kArchiveSize);
  delete[] delta_fit;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T>
IndividualSnapshot<T> Population<T>::SnapshotIndividual(int index) {
  IndividualSnapshot<T> snapshot;
  snapshot.pop.assign(Nvar, 0);
  snapshot.newpop.assign(Nvar, 0);
  snapshot.pop_fit = pop_fit[index];
  snapshot.newpop_fit = newpop_fit[index];

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int j = 0; j < Nvar; j++) {
    snapshot.pop[j] = pop[index][j];
    snapshot.newpop[j] = newpop[index][j];
  }

  // 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
  return snapshot;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T>
void Population<T>::RestorePopIndividual(
    int index, const IndividualSnapshot<T>& snapshot) {
  pop_fit[index] = snapshot.pop_fit;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int j = 0; j < Nvar; j++)
    pop[index][j] = snapshot.pop[j];
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> T** Population<T>::CreateMatrix(int nRow, int nCol) {
  T** ppT = new T*[nRow];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int iRow = 0; iRow < nRow; iRow++) {
    ppT[iRow] = new T[nCol];
  }
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return ppT;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> void Population<T>::DeleteMatrix(T** ppT, int nRow) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (ppT == nullptr)
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int iRow = 0; iRow < nRow; iRow++) {
    delete[] ppT[iRow];
  }
  delete[] ppT;
  ppT = NULL;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> double Population<T>::randval(double low, double high) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return ((double)(rand() % 1000) / 1000.0) * (high - low) + low;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T>
double Population<T>::pop_random_variance(int p_start, int p_end) {
  int bit = rand() % Nvar;
  double Bave = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++)
    Bave += pop[i][bit];
  Bave /= Popsize;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  double variance = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = p_start; i < p_end; i++)
    variance += pow((pop[i][bit] - Bave), 2.0);
  variance = sqrt(variance) / Popsize;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return variance;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T>
double Population<T>::pop_random_entropy(int subn, int p_start, int p_end) {
  double entr = 0;
  int reg = 0;
  int bit = rand() % Nvar;
  double* pp = new double[subn];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < subn; i++)
    pp[i] = 0;

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = p_start; i < p_end; i++) {
    reg = subn * (pop[i][bit] - Lbound) / (Ubound - Lbound);
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (reg >= subn)
      reg = subn - 1;
    pp[reg]++;
  }

  // 段落说明：遍历当前任务、变量、个体或动作集合，逐项完成本段更新。
  for (int i = 0; i < subn; i++)
    pp[i] /= subn;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < subn; i++) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (pp[i] != 0)
      entr -= pp[i] * log(pp[i]);
  }
  delete[] pp;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return entr;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T>
void Population<T>::heap_adjust(T** num, int s, int len, int cbit) {
  T* temp = num[s];
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 2 * s + 1; i < len; i = 2 * i + 1) {
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (i < (len - 1) && num[i][cbit] < num[i + 1][cbit])
      i++;
    // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
    if (temp[cbit] > num[i][cbit])
      break;
    num[s] = num[i];
    s = i;
  }
  num[s] = temp;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> void Population<T>::heap_sort(T** num, int len, int cbit) {
  int i;
  T* temp;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = len / 2 - 1; i >= 0; i--)
    heap_adjust(num, i, len, cbit);
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = len - 1; i > 0; i--) {
    temp = num[0];
    num[0] = num[i];
    num[i] = temp;
    heap_adjust(num, 0, i, cbit);
  }
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> void Population<T>::swap(T& x, T& y) {
  T temp;
  temp = x;
  x = y;
  y = temp;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> void Population<T>::worst_and_best() {
  int i;
  cur_best = 0;
  cur_worst = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = 0; i < Popsize; i++) {
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (pop_fit[i] < pop_fit[cur_best])
      cur_best = i;
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    else if (pop_fit[i] > pop_fit[cur_worst])
      cur_worst = i;
  }
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> void Population<T>::Elist() {
  int i, j;
  double delta_f;
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (pop_fit[cur_best] < gbest_fit) {
    delta_f = pop_fit[cur_best] - gbest_fit;
    std::memcpy(gbest, pop[cur_best], sizeof(T) * Nvar);
    gbest_fit = pop_fit[cur_best];
    // 控制说明：依据目标值决定接受、最优更新或审计路径。
    if (delta_pop == nullptr || delta_fit == nullptr)
      // 控制说明：返回本阶段计算结果或状态码给调用方。
      return;
    i = 0;
    // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
    while (i < kArchiveSize) {
      // 控制说明：依据目标值决定接受、最优更新或审计路径。
      if (delta_f > delta_fit[i]) {
        delta_update_count++;
        // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
        for (j = 0; j < Nvar; j++) {
          delta_pop[i][j] = gbest[j];
          delta_pop[i][Nvar + j] = pop[cur_best][j] - gbest[j];
        }
        break;
      } else
        i++;
    }
  } else {
    std::memcpy(pop[cur_worst], gbest, sizeof(T) * Nvar);
    pop_fit[cur_worst] = gbest_fit;
  }
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> inline double Population<T>::average_fit() {
  double ave = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (int i = 0; i < Popsize; i++)
    ave += pop_fit[i];
  ave /= Popsize;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return ave;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> void Population<T>::CRfit() {
  CRold = CRnew;
  double ave = average_fit() / pop_fit[cur_worst];
  double best = pop_fit[cur_best] / pop_fit[cur_worst];
  // 控制说明：依据目标值决定接受、最优更新或审计路径。
  if (best == 1)
    CRnew = 1;
  // 控制说明：条件不成立时执行互斥的备用处理路径。
  else
    CRnew = (1 - ave) / (1 - best);
}

// template<class T>
// double Population<T>::maxElement(const double* rgd, int len)
//{
//   return *std::max_element(rgd, rgd + len);
// }
//
// template<class T>
// double Population<T>::minElement(const double* rgd, int len)
//{
//   return *std::min_element(rgd, rgd + len);
// }

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T> double Population<T>::square(double d) {
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return d * d;
}

/**
 * @return (0,1)-normally distributed random number
 */
template <class T> double Population<T>::gauss(void) {
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (stored) {
    stored = false;
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return hold;
  }
  stored = true;
  double x1, x2, rquad;
  // 控制说明：重复执行该步骤，直到预算、收敛或合法性条件满足。
  do {
    x1 = 2.0 * uniform() - 1.0;
    x2 = 2.0 * uniform() - 1.0;
    rquad = x1 * x1 + x2 * x2;
  } while (rquad >= 1 || rquad <= 0);
  const double fac = sqrt((-2) * log(rquad) / rquad);
  hold = fac * x1;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return fac * x2;
}
/**
 * @return (0,1)-uniform distributed random number
 */
template <class T> double Population<T>::uniform(void) {
  long tmp = aktseed / 127773;
  aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (aktseed < 0)
    aktseed += 2147483647;
  tmp = aktrand / 67108865;
  aktrand = rgrand[tmp];
  rgrand[tmp] = aktseed;
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return (double)aktrand / 2.147483647e9;
}
/** sqrt(a^2 + b^2) numerically stable. */
template <class T> double Population<T>::myhypot(double a, double b) {
  const double fabsa = fabs(a), fabsb = fabs(b);
  // 控制说明：检查本分支的配置、边界或状态条件，再执行对应处理。
  if (fabsa > fabsb) {
    const double r = b / a;
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return fabsa * sqrt(1. + r * r);
  } else if (b != 0.) {
    const double r = a / b;
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return fabsb * sqrt(1. + r * r);
  } else
    // 控制说明：返回本阶段计算结果或状态码给调用方。
    return 0.;
}

// 段落说明：定义本模块使用的类型、状态或配置数据结构。
template <class T>
double Population<T>::Euclidean_dis(double* a, double* b, int len) {
  int i = 0;
  double dis = 0;
  // 控制说明：遍历该集合并对每个元素执行下面的同一更新规则。
  for (i = 0; i < len; i++) {
    dis += fabs(a[i] - b[i]);
  }
  dis = sqrt(dis);
  // 控制说明：返回本阶段计算结果或状态码给调用方。
  return dis;
}

// 段落说明：执行当前逻辑段的数据变换、边界处理或状态更新。
#endif
