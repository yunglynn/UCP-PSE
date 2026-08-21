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

template <class T> struct IndividualSnapshot {
  vector<T> pop;
  vector<T> newpop;
  double pop_fit;
  double newpop_fit;
};

template <class T> class Population {
public:
  Population(int psize, int nn, T lb, T ub);
  virtual ~Population();

public:
  T** pop;
  double* pop_fit;
  T** newpop;
  double* newpop_fit;
  T* gbest;
  double gbest_fit;

  double** delta_pop;
  double* delta_fit;
  int delta_update_count;

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

  long int t;
  long unsigned seed;
  long int aktseed;
  long int aktrand;
  long int rgrand[32];

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
  for (int i = 0; i < Popsize; i++)
    for (int j = 0; j < Nvar; j++) {
      pop[i][j] = (T)randval(Lbound, Ubound);
    }

  for (int i = 0; i < Popsize; i++) {
    for (int j = 0; j < Nvar; j++) {
      newpop[i][j] = (T)randval(Lbound, Ubound);
    }
  }
  for (int i = 0; i < Nvar; i++) {
    gbest[i] = (T)randval(Lbound, Ubound);
  }

  delta_pop = nullptr;
  delta_fit = nullptr;
  if (Nvar < 10000000) {
    delta_pop = new double*[kArchiveSize];
    delta_fit = new double[kArchiveSize];
    for (int i = 0; i < kArchiveSize; i++)
      delta_pop[i] = new double[Nvar * 2];
    for (int i = 0; i < kArchiveSize; i++)
      delta_fit[i] = 0;
  }
  delta_update_count = 0;

  // CMAES

  t = rand() + 1;
  seed = (long unsigned)(t < 0 ? -t : t);
  if (seed < 1)
    seed = 1;
  aktseed = seed;
  for (int i = 39; i >= 0; --i) {
    long tmp = aktseed / 127773;
    aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
    if (aktseed < 0)
      aktseed += 2147483647;
    if (i < 32)
      rgrand[i] = aktseed;
  }
  aktrand = rgrand[0];
}

template <class T> Population<T>::~Population() {
  DeleteMatrix(pop, Popsize);
  DeleteMatrix(newpop, Popsize);
  delete[] gbest;
  delete[] pop_fit;
  delete[] newpop_fit;
  DeleteMatrix(delta_pop, kArchiveSize);
  delete[] delta_fit;
}

template <class T>
IndividualSnapshot<T> Population<T>::SnapshotIndividual(int index) {
  IndividualSnapshot<T> snapshot;
  snapshot.pop.assign(Nvar, 0);
  snapshot.newpop.assign(Nvar, 0);
  snapshot.pop_fit = pop_fit[index];
  snapshot.newpop_fit = newpop_fit[index];

  for (int j = 0; j < Nvar; j++) {
    snapshot.pop[j] = pop[index][j];
    snapshot.newpop[j] = newpop[index][j];
  }

  return snapshot;
}

template <class T>
void Population<T>::RestorePopIndividual(
    int index, const IndividualSnapshot<T>& snapshot) {
  pop_fit[index] = snapshot.pop_fit;
  for (int j = 0; j < Nvar; j++)
    pop[index][j] = snapshot.pop[j];
}

template <class T> T** Population<T>::CreateMatrix(int nRow, int nCol) {
  T** ppT = new T*[nRow];
  for (int iRow = 0; iRow < nRow; iRow++) {
    ppT[iRow] = new T[nCol];
  }
  return ppT;
}

template <class T> void Population<T>::DeleteMatrix(T** ppT, int nRow) {
  if (ppT == nullptr)
    return;
  for (int iRow = 0; iRow < nRow; iRow++) {
    delete[] ppT[iRow];
  }
  delete[] ppT;
  ppT = NULL;
}

template <class T> double Population<T>::randval(double low, double high) {
  return ((double)(rand() % 1000) / 1000.0) * (high - low) + low;
}

template <class T>
double Population<T>::pop_random_variance(int p_start, int p_end) {
  int bit = rand() % Nvar;
  double Bave = 0;
  for (int i = p_start; i < p_end; i++)
    Bave += pop[i][bit];
  Bave /= Popsize;

  double variance = 0;
  for (int i = p_start; i < p_end; i++)
    variance += pow((pop[i][bit] - Bave), 2.0);
  variance = sqrt(variance) / Popsize;
  return variance;
}

template <class T>
double Population<T>::pop_random_entropy(int subn, int p_start, int p_end) {
  double entr = 0;
  int reg = 0;
  int bit = rand() % Nvar;
  double* pp = new double[subn];
  for (int i = 0; i < subn; i++)
    pp[i] = 0;

  for (int i = p_start; i < p_end; i++) {
    reg = subn * (pop[i][bit] - Lbound) / (Ubound - Lbound);
    if (reg >= subn)
      reg = subn - 1;
    pp[reg]++;
  }

  for (int i = 0; i < subn; i++)
    pp[i] /= subn;
  for (int i = 0; i < subn; i++) {
    if (pp[i] != 0)
      entr -= pp[i] * log(pp[i]);
  }
  delete[] pp;
  return entr;
}

template <class T>
void Population<T>::heap_adjust(T** num, int s, int len, int cbit) {
  T* temp = num[s];
  for (int i = 2 * s + 1; i < len; i = 2 * i + 1) {
    if (i < (len - 1) && num[i][cbit] < num[i + 1][cbit])
      i++;
    if (temp[cbit] > num[i][cbit])
      break;
    num[s] = num[i];
    s = i;
  }
  num[s] = temp;
}

template <class T> void Population<T>::heap_sort(T** num, int len, int cbit) {
  int i;
  T* temp;
  for (i = len / 2 - 1; i >= 0; i--)
    heap_adjust(num, i, len, cbit);
  for (i = len - 1; i > 0; i--) {
    temp = num[0];
    num[0] = num[i];
    num[i] = temp;
    heap_adjust(num, 0, i, cbit);
  }
}

template <class T> void Population<T>::swap(T& x, T& y) {
  T temp;
  temp = x;
  x = y;
  y = temp;
}

template <class T> void Population<T>::worst_and_best() {
  int i;
  cur_best = 0;
  cur_worst = 0;
  for (i = 0; i < Popsize; i++) {
    if (pop_fit[i] < pop_fit[cur_best])
      cur_best = i;
    else if (pop_fit[i] > pop_fit[cur_worst])
      cur_worst = i;
  }
}

template <class T> void Population<T>::Elist() {
  int i, j;
  double delta_f;
  if (pop_fit[cur_best] < gbest_fit) {
    delta_f = pop_fit[cur_best] - gbest_fit;
    std::memcpy(gbest, pop[cur_best], sizeof(T) * Nvar);
    gbest_fit = pop_fit[cur_best];
    if (delta_pop == nullptr || delta_fit == nullptr)
      return;
    i = 0;
    while (i < kArchiveSize) {
      if (delta_f > delta_fit[i]) {
        delta_update_count++;
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

template <class T> inline double Population<T>::average_fit() {
  double ave = 0;
  for (int i = 0; i < Popsize; i++)
    ave += pop_fit[i];
  ave /= Popsize;
  return ave;
}

template <class T> void Population<T>::CRfit() {
  CRold = CRnew;
  double ave = average_fit() / pop_fit[cur_worst];
  double best = pop_fit[cur_best] / pop_fit[cur_worst];
  if (best == 1)
    CRnew = 1;
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

template <class T> double Population<T>::square(double d) {
  return d * d;
}

/**
 * @return (0,1)-normally distributed random number
 */
template <class T> double Population<T>::gauss(void) {
  if (stored) {
    stored = false;
    return hold;
  }
  stored = true;
  double x1, x2, rquad;
  do {
    x1 = 2.0 * uniform() - 1.0;
    x2 = 2.0 * uniform() - 1.0;
    rquad = x1 * x1 + x2 * x2;
  } while (rquad >= 1 || rquad <= 0);
  const double fac = sqrt((-2) * log(rquad) / rquad);
  hold = fac * x1;
  return fac * x2;
}
/**
 * @return (0,1)-uniform distributed random number
 */
template <class T> double Population<T>::uniform(void) {
  long tmp = aktseed / 127773;
  aktseed = 16807 * (aktseed - tmp * 127773) - 2836 * tmp;
  if (aktseed < 0)
    aktseed += 2147483647;
  tmp = aktrand / 67108865;
  aktrand = rgrand[tmp];
  rgrand[tmp] = aktseed;
  return (double)aktrand / 2.147483647e9;
}
/** sqrt(a^2 + b^2) numerically stable. */
template <class T> double Population<T>::myhypot(double a, double b) {
  const double fabsa = fabs(a), fabsb = fabs(b);
  if (fabsa > fabsb) {
    const double r = b / a;
    return fabsa * sqrt(1. + r * r);
  } else if (b != 0.) {
    const double r = a / b;
    return fabsb * sqrt(1. + r * r);
  } else
    return 0.;
}

template <class T>
double Population<T>::Euclidean_dis(double* a, double* b, int len) {
  int i = 0;
  double dis = 0;
  for (i = 0; i < len; i++) {
    dis += fabs(a[i] - b[i]);
  }
  dis = sqrt(dis);
  return dis;
}

#endif
