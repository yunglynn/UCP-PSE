/* UCP-PSE orchestration: protected defaults, MPI/local archive setup, sparse
 * search loop, ring migration, timing, and exact final verification.  Candidate
 * generation is in MultimethodMeme.cpp; evaluation is in Problems.cpp. */
#include "Config.h"
#include "Multimethod.h"
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <chrono>
#include <iostream>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif
using namespace std;
#ifndef MIGRATION_INTERVAL
#define MIGRATION_INTERVAL 10
#endif
#ifndef LOCAL_ARCHIVE_ENABLED
#define LOCAL_ARCHIVE_ENABLED 1
#endif
#ifndef STRICT_SINGLE_NO_MEMORY
#define STRICT_SINGLE_NO_MEMORY 0
#endif
#ifndef SURROGATE_ENABLED
#define SURROGATE_ENABLED 1
#endif
#ifndef MEME_SEARCH_MODE
#define MEME_SEARCH_MODE 0
#endif
#ifndef MPI_INDEPENDENT_SEED
#define MPI_INDEPENDENT_SEED 0
#endif
#ifndef SERIAL_POPULATION_ABLATION
#define SERIAL_POPULATION_ABLATION 0
#endif
#ifndef BASE_SEED
#define BASE_SEED 20260616
#endif
#ifndef GLOBAL_SEARCH_ALG
#ifdef USE_MPI
#define GLOBAL_SEARCH_ALG ((int)SearchAlg::ADE)
#else
#define GLOBAL_SEARCH_ALG 0
#endif
#endif
#ifndef GLOBAL_SEARCH_ENABLED
#define GLOBAL_SEARCH_ENABLED 0
#endif
#ifndef DTGP_TASK_PRIOR
#define DTGP_TASK_PRIOR 0
#endif
#ifndef DTGP_TASK_PRIOR_ANCHOR_ONLY
#define DTGP_TASK_PRIOR_ANCHOR_ONLY 0
#endif
#ifndef TASK_PRIOR_PROXY_DESIGN
#define TASK_PRIOR_PROXY_DESIGN 0
#endif
#ifndef TRI_POLICY_RANDOM_ABLATION
#define TRI_POLICY_RANDOM_ABLATION 0
#endif
#ifndef ROTATING_TRUE_AUDIT_INTERVAL
#if TNUM >= 1000 && TNUM <= 100000
#define ROTATING_TRUE_AUDIT_INTERVAL 10
#else
#define ROTATING_TRUE_AUDIT_INTERVAL 0
#endif
#endif
#ifndef AUDIT_STRUCTURED_MEME
#if TNUM >= 1000 && TNUM <= 100000
#define AUDIT_STRUCTURED_MEME 1
#else
#define AUDIT_STRUCTURED_MEME 0
#endif
#endif

#if DTGP_TASK_PRIOR || TASK_PRIOR_PROXY_DESIGN
static double prior_fractional(double value) {
  return value - std::floor(value);
}

static void construct_task_prior(MultiMet& solver, int slot, int rule) {
  const int tasks = solver.CE_Tnum;
  const int operations = solver.M_Jnum * solver.M_OPTnum;
  double* x = solver.pop[slot];
  for (int i = 0; i < tasks; ++i) {
    const double comp = solver.CETask_Property[i].Computation / 199.0;
    const double comm = solver.CETask_Property[i].Communication / 4999.0;
    const double relation =
        (solver.CETask_Property[i].Precedence.size() +
         solver.CETask_Property[i].Interact.size() +
         solver.CETask_Property[i].Start_Pre.size() +
         solver.CETask_Property[i].End_Pre.size()) / 8.0;
    const double position = static_cast<double>(i) / std::max(1, tasks - 1);
    switch (rule) {
      case 0: x[i] = 0.0; break;
      case 1: x[i] = 1.0; break;
      case 2: x[i] = comp > comm ? 0.0 : 1.0; break;
      case 3: x[i] = comp + relation > comm + 0.5 ? 0.0 : 1.0; break;
      case 4: x[i] = 1.0 / (1.0 + std::exp(-4.0 * (comm - comp))); break;
      case 5: x[i] = 1.0 / (1.0 + std::exp(-4.0 * (comp - comm))); break;
      case 6: x[i] = 1.0 / (1.0 + std::exp(-3.0 * (comm + relation - comp))); break;
      default: x[i] = ((i + rule) & 1) ? 1.0 : 0.0; break;
    }
    const double resource = rule % 2 == 0
                                ? comp + 0.5 * comm + relation
                                : comm + 0.5 * comp - relation;
    x[tasks + i] = prior_fractional(resource + 0.61803398875 * position +
                                    0.137 * rule);
  }
  for (int i = 0; i < operations; ++i) {
    const double duration = solver.MTask_Time[i] / 300.0;
    const double position = static_cast<double>(i) /
                            std::max(1, operations - 1);
    const double stage = static_cast<double>(i % solver.M_OPTnum) /
                         std::max(1, solver.M_OPTnum - 1);
    double sequence = position;
    if (rule == 1) sequence = duration;
    else if (rule == 2) sequence = 0.75 * stage + 0.25 * position;
    else if (rule == 3) sequence = 0.75 * stage + 0.25 * duration;
    else if (rule == 4) sequence = duration + stage;
    else if (rule == 5) sequence = stage - duration;
    else if (rule == 6) sequence = duration + 0.5 * position + stage;
    else if (rule == 7) sequence = 1.0 - duration + stage;
    x[2 * tasks + i] = prior_fractional(sequence);
    x[2 * tasks + operations + i] = prior_fractional(
        duration + 0.38196601125 * position + 0.173 * rule);
  }
  const size_t bytes = sizeof(double) * solver.Nvar;
  std::memcpy(solver.newpop[slot], x, bytes);
  std::memcpy(solver.ibest[slot], x, bytes);
  const double fit = solver.EvaluFunc(
      x, solver.Cnum, solver.Enum, solver.Dnum, solver.CE_Tnum,
      solver.M_Jnum, solver.M_OPTnum, solver.CETask_Property,
      solver.MTask_Time, solver.EtoD_Distance, solver.DtoD_Distance,
      solver.AvailDeviceList, solver.EnergyList, solver.CloudDevices,
      solver.EdgeDevices, solver.CloudLoad, solver.EdgeLoad,
      solver.DeviceLoad, solver.CETask_coDevice, solver.Edge_Device_comm,
      solver.ST, solver.ET, solver.CE_ST, solver.CE_ET);
  solver.pop_fit[slot] = solver.newpop_fit[slot] = solver.ibest_fit[slot] = fit;
}

#if TASK_PRIOR_PROXY_DESIGN
static void apply_task_prior_proxy_design(MultiMet& solver) {
  const int slot = 0;
  vector<double> anchor(solver.Nvar);
  double anchor_fit = 1e300;
  for (int rule = 0; rule < 8; ++rule) {
    construct_task_prior(solver, slot, rule);
    if (solver.pop_fit[slot] < anchor_fit) {
      anchor_fit = solver.pop_fit[slot];
      std::memcpy(anchor.data(), solver.pop[slot],
                  sizeof(double) * solver.Nvar);
    }
  }
  std::memcpy(solver.pop[slot], anchor.data(), sizeof(double) * solver.Nvar);
  std::memcpy(solver.newpop[slot], anchor.data(),
              sizeof(double) * solver.Nvar);
  std::memcpy(solver.ibest[slot], anchor.data(), sizeof(double) * solver.Nvar);
  std::memcpy(solver.gbest, anchor.data(), sizeof(double) * solver.Nvar);
  solver.pop_fit[slot] = anchor_fit;
  solver.newpop_fit[slot] = anchor_fit;
  solver.ibest_fit[slot] = anchor_fit;
  solver.gbest_fit = anchor_fit;
  solver.meme_last_best = anchor_fit;
  solver.worst_and_best();
}
#endif

#if DTGP_TASK_PRIOR
static void apply_task_priors(MultiMet& solver, int rank) {
#if DTGP_TASK_PRIOR_ANCHOR_ONLY
  if (rank != 0)
    return;
  const int slots = solver.Popsize;
  const size_t vector_size = static_cast<size_t>(slots) * solver.Nvar;
  vector<double> saved_pop(vector_size), saved_newpop(vector_size);
  vector<double> saved_ibest(vector_size);
  vector<double> saved_pop_fit(solver.pop_fit, solver.pop_fit + slots);
  vector<double> saved_newpop_fit(solver.newpop_fit,
                                  solver.newpop_fit + slots);
  vector<double> saved_ibest_fit(solver.ibest_fit,
                                 solver.ibest_fit + slots);
  for (int i = 0; i < slots; ++i) {
    std::memcpy(saved_pop.data() + static_cast<size_t>(i) * solver.Nvar,
                solver.pop[i], sizeof(double) * solver.Nvar);
    std::memcpy(saved_newpop.data() + static_cast<size_t>(i) * solver.Nvar,
                solver.newpop[i], sizeof(double) * solver.Nvar);
    std::memcpy(saved_ibest.data() + static_cast<size_t>(i) * solver.Nvar,
                solver.ibest[i], sizeof(double) * solver.Nvar);
  }
  vector<double> anchor(solver.Nvar);
  double anchor_fit = 1e300;
  for (int slot = 0; slot < slots; ++slot) {
    construct_task_prior(solver, slot, slot % 8);
    // Select the initialization anchor by the exact scheduling objective.
    // The hybrid surrogate is still updated by construct_task_prior(), but it
    // is not yet sufficiently trained to rank the initial prior candidates.
    const double exact_fit = CED_Schedule(
        solver.pop[slot], solver.Cnum, solver.Enum, solver.Dnum,
        solver.CE_Tnum, solver.M_Jnum, solver.M_OPTnum,
        solver.CETask_Property, solver.MTask_Time, solver.EtoD_Distance,
        solver.DtoD_Distance, solver.AvailDeviceList, solver.EnergyList,
        solver.CloudDevices, solver.EdgeDevices, solver.CloudLoad,
        solver.EdgeLoad, solver.DeviceLoad, solver.CETask_coDevice,
        solver.Edge_Device_comm, solver.ST, solver.ET, solver.CE_ST,
        solver.CE_ET);
    if (exact_fit < anchor_fit) {
      anchor_fit = exact_fit;
      std::memcpy(anchor.data(), solver.pop[slot],
                  sizeof(double) * solver.Nvar);
    }
  }
  for (int i = 0; i < slots; ++i) {
    std::memcpy(solver.pop[i],
                saved_pop.data() + static_cast<size_t>(i) * solver.Nvar,
                sizeof(double) * solver.Nvar);
    std::memcpy(solver.newpop[i],
                saved_newpop.data() + static_cast<size_t>(i) * solver.Nvar,
                sizeof(double) * solver.Nvar);
    std::memcpy(solver.ibest[i],
                saved_ibest.data() + static_cast<size_t>(i) * solver.Nvar,
                sizeof(double) * solver.Nvar);
    solver.pop_fit[i] = saved_pop_fit[i];
    solver.newpop_fit[i] = saved_newpop_fit[i];
    solver.ibest_fit[i] = saved_ibest_fit[i];
  }
  solver.worst_and_best();
  std::memcpy(solver.gbest, anchor.data(), sizeof(double) * solver.Nvar);
  solver.gbest_fit = anchor_fit;
  solver.meme_last_best = solver.gbest_fit;
#else
  if (solver.Popsize == 1) {
    construct_task_prior(solver, 0, rank % 8);
    const double exact_fit = CED_Schedule(
        solver.pop[0], solver.Cnum, solver.Enum, solver.Dnum,
        solver.CE_Tnum, solver.M_Jnum, solver.M_OPTnum,
        solver.CETask_Property, solver.MTask_Time, solver.EtoD_Distance,
        solver.DtoD_Distance, solver.AvailDeviceList, solver.EnergyList,
        solver.CloudDevices, solver.EdgeDevices, solver.CloudLoad,
        solver.EdgeLoad, solver.DeviceLoad, solver.CETask_coDevice,
        solver.Edge_Device_comm, solver.ST, solver.ET, solver.CE_ST,
        solver.CE_ET);
    solver.pop_fit[0] = solver.newpop_fit[0] = solver.ibest_fit[0] = exact_fit;
  } else if (rank == 0) {
    for (int slot = 0; slot < solver.Popsize; ++slot) {
      construct_task_prior(solver, slot, slot % 8);
      const double exact_fit = CED_Schedule(
          solver.pop[slot], solver.Cnum, solver.Enum, solver.Dnum,
          solver.CE_Tnum, solver.M_Jnum, solver.M_OPTnum,
          solver.CETask_Property, solver.MTask_Time, solver.EtoD_Distance,
          solver.DtoD_Distance, solver.AvailDeviceList, solver.EnergyList,
          solver.CloudDevices, solver.EdgeDevices, solver.CloudLoad,
          solver.EdgeLoad, solver.DeviceLoad, solver.CETask_coDevice,
          solver.Edge_Device_comm, solver.ST, solver.ET, solver.CE_ST,
          solver.CE_ET);
      solver.pop_fit[slot] = solver.newpop_fit[slot] =
          solver.ibest_fit[slot] = exact_fit;
    }
  } else {
    return;
  }
  solver.worst_and_best();
  std::memcpy(solver.gbest, solver.pop[solver.cur_best],
              sizeof(double) * solver.Nvar);
  solver.gbest_fit = solver.pop_fit[solver.cur_best];
  solver.meme_last_best = solver.gbest_fit;
#endif
}
#endif
#endif
static int proxy_prewarm_generations(int population_size) {
#if TNUM >= 1000000
  // Complete the eight-point lightweight surrogate design before timing.
  return 8;
#endif
#ifdef USE_MPI
  return std::max(0,
                  static_cast<int>(std::ceil(std::log2(population_size))) - 1);
#else
  (void)population_size;
  return 0;
#endif
}

#ifdef USE_MPI
static void rank_range(int rank, int size, int total, int& start, int& end) {
  start = rank * total / size;
  end = (rank + 1) * total / size;
}

static void sync_initial_population(MultiMet& solver) {
  for (int i = 0; i < solver.Popsize; i++) {
    MPI_Bcast(solver.pop[i], solver.Nvar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(solver.newpop[i], solver.Nvar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(solver.ibest[i], solver.Nvar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (solver.velocity != nullptr)
      MPI_Bcast(solver.velocity[i], solver.Nvar, MPI_DOUBLE, 0,
                MPI_COMM_WORLD);
  }

  MPI_Bcast(solver.pop_fit, solver.Popsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(solver.newpop_fit, solver.Popsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(solver.ibest_fit, solver.Popsize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(solver.gbest, solver.Nvar, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&solver.gbest_fit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  solver.worst_and_best();
}

#endif

#ifdef USE_MPI
static void inject_solution(MultiMet& solver, vector<double>& candidate,
                            int active_slot) {
  const double fit = candidate[solver.Nvar];
#if LOCAL_ARCHIVE_ENABLED
  solver.worst_and_best();
  if (fit >= solver.pop_fit[solver.cur_worst])
    return;
  const int target = solver.cur_worst;
#else
  if (fit >= solver.newpop_fit[active_slot])
    return;
  const int target = active_slot;
#endif
  for (int j = 0; j < solver.Nvar; j++) {
    solver.pop[target][j] = candidate[j];
    solver.newpop[target][j] = candidate[j];
    solver.ibest[target][j] = candidate[j];
  }
  solver.pop_fit[target] = fit;
  solver.newpop_fit[target] = fit;
  solver.ibest_fit[target] = fit;

  if (fit < solver.gbest_fit) {
    for (int j = 0; j < solver.Nvar; j++)
      solver.gbest[j] = candidate[j];
    solver.gbest_fit = fit;
  }
  solver.worst_and_best();
}

static void pack_migration_solution(MultiMet& solver, int active_slot,
                                    vector<double>& payload) {
  int payload_size = solver.Nvar + 1;
  if ((int)payload.size() != payload_size)
    payload.resize(payload_size);
#if LOCAL_ARCHIVE_ENABLED
  for (int j = 0; j < solver.Nvar; ++j)
    payload[j] = solver.gbest[j];
  payload[solver.Nvar] = solver.gbest_fit;
#else
  for (int j = 0; j < solver.Nvar; j++)
    payload[j] = solver.newpop[active_slot][j];
  payload[solver.Nvar] = solver.newpop_fit[active_slot];
#endif
}

static void sendrecv_best_solution(MultiMet& solver, int send_to, int recv_from,
                                   int active_slot, int tag) {
  int payload_size = solver.Nvar + 1;
  static vector<double> outgoing;
  static vector<double> incoming;
  pack_migration_solution(solver, active_slot, outgoing);
  if ((int)incoming.size() != payload_size)
    incoming.resize(payload_size);
  MPI_Sendrecv(outgoing.data(), payload_size, MPI_DOUBLE, send_to, tag,
               incoming.data(), payload_size, MPI_DOUBLE, recv_from, tag,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  inject_solution(solver, incoming, active_slot);
}

static void migrate_best_solution(MultiMet& solver, int rank, int mpi_size,
                                  int active_slot) {
  if (mpi_size <= 1)
    return;

  const int send_to = (rank + 1) % mpi_size;
  const int recv_from = (rank - 1 + mpi_size) % mpi_size;
  sendrecv_best_solution(solver, send_to, recv_from, active_slot, 1001);
}

#endif

static unsigned make_seed(int rank) {
  unsigned seed = BASE_SEED;
  const char* seed_env = getenv("CED_SEED");
  if (seed_env != NULL)
    seed = (unsigned)strtoul(seed_env, NULL, 10);
#if defined(USE_MPI) && MPI_INDEPENDENT_SEED
  seed += rank * 1000003u;
#elif defined(USE_MPI) && TNUM >= 1000000
  // The million-scale layout stores one local individual per MPI rank.
  seed += rank * 1000003u;
#else
  (void)rank;
#endif
  return seed;
}

static double current_time() {
  using Clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

#ifdef USE_MPI
static bool sync_before_next_generation(int generation, int interval) {
  return interval > 0 && generation + 1 < MAXGEN &&
         (generation + 1) % interval == 0;
}
#endif

static int env_int_or_default(const char* name, int default_value) {
  const char* value = getenv(name);
  if (value == NULL || value[0] == '\0')
    return default_value;
  return atoi(value);
}

static int configured_meme_search_mode() {
  return env_int_or_default("CED_MEME_SEARCH_MODE", MEME_SEARCH_MODE);
}

static bool configured_global_search_enabled() {
  return env_int_or_default("CED_GLOBAL_SEARCH_ENABLED",
                            GLOBAL_SEARCH_ENABLED) != 0;
}

static int rank_global_search_alg(int rank) {
  const char* algorithm_env = getenv("CED_GLOBAL_SEARCH_ALG");
  if (algorithm_env != NULL && algorithm_env[0] != '\0')
    return atoi(algorithm_env);
  (void)rank;
  return GLOBAL_SEARCH_ALG;
}

static void run_global_search(MultiMet& solver, int generation, int p_start,
                              int p_end, int rank) {
  solver.RunGlobalSearch(rank_global_search_alg(rank), generation, MAXGEN,
                         p_start, p_end);
}

static double meme_search_scale(const MultiMet& solver) {
  const int encoded_choices_per_job =
      solver.M_OPTnum * solver.M_OPTnum * (MultiMet::MemePolicyActionCount / 2);
  return 1.0 / std::sqrt(static_cast<double>(encoded_choices_per_job));
}

static void run_meme_search(MultiMet& solver, int generation, int p_start,
                            int p_end) {
#ifdef USE_MPI
  solver.RunMemeSearchStrategy(configured_meme_search_mode(), generation,
                               MAXGEN, meme_search_scale(solver), p_start,
                               p_end);
#else
  (void)p_start;
  (void)p_end;
  solver.RunMemeSearchStrategy(configured_meme_search_mode(), generation,
                               MAXGEN, meme_search_scale(solver), 0,
                               solver.Popsize);
#endif
}

#if defined(USE_MPI) && LOCAL_ARCHIVE_ENABLED
static void seed_local_working_slot_from_archive(MultiMet& solver, int p_start,
                                                 int p_end) {
  if (p_start >= p_end)
    return;

  int archive_best = 0;
  for (int i = 1; i < solver.Popsize; ++i)
    if (solver.ibest_fit[i] < solver.ibest_fit[archive_best])
      archive_best = i;

  const size_t bytes = sizeof(double) * static_cast<size_t>(solver.Nvar);
  for (int i = p_start; i < p_end; ++i) {
    std::memcpy(solver.newpop[i], solver.ibest[archive_best], bytes);
    solver.newpop_fit[i] = solver.ibest_fit[archive_best];
  }
}
#endif

static void print_progress(int rank, int generation, double current_best,
                           double current_average) {
  if (rank != 0 || !VERBOSE_OUTPUT)
    return;

  cout << "Generation " << generation << ", current best = " << current_best
       << ", average = " << current_average << endl;
}

static void print_summary(int rank, int generation, double best_fit,
                          double elapsed_time) {
  if (rank != 0)
    return;

  cout << "Generation = " << generation << endl;
  cout << "The best solution = " << best_fit << endl;
  cout << "Time = " << elapsed_time << " s" << endl;
}

static double true_schedule_value(MultiMet& solver, double* candidate) {
  return CED_Schedule(
      candidate, CNUM, ENUM, DNUM, TNUM, TNUM, MOPT_NUM, solver.CETask_Property,
      solver.MTask_Time, solver.EtoD_Distance, solver.DtoD_Distance,
      solver.AvailDeviceList, solver.EnergyList, solver.CloudDevices,
      solver.EdgeDevices, solver.CloudLoad, solver.EdgeLoad, solver.DeviceLoad,
      solver.CETask_coDevice, solver.Edge_Device_comm, solver.ST, solver.ET,
      solver.CE_ST, solver.CE_ET);
}

static bool seen_solution_candidate(vector<const double*>& seen,
                                    const double* candidate, int nvar) {
  const size_t bytes = sizeof(double) * static_cast<size_t>(nvar);
  for (const double* value : seen) {
    if (value == candidate || std::memcmp(value, candidate, bytes) == 0)
      return true;
  }
  seen.push_back(candidate);
  return false;
}

static void update_verified_best(MultiMet& solver, double* candidate,
                                 vector<const double*>& seen,
                                 double& best_value,
                                 vector<double>* best_candidate) {
  if (seen_solution_candidate(seen, candidate, solver.Nvar))
    return;
  double value = true_schedule_value(solver, candidate);
  if (value < best_value) {
    best_value = value;
    if (best_candidate != nullptr)
      best_candidate->assign(candidate, candidate + solver.Nvar);
  }
}

static double verify_final_candidates_true(MultiMet& solver, int p_start,
                                           int p_end,
                                           vector<double>* best_candidate = nullptr) {
  vector<const double*> seen;
#if LOCAL_ARCHIVE_ENABLED
  seen.reserve(1 + 3 * (p_end - p_start));
  double best_value = 1e300;
  update_verified_best(solver, solver.gbest, seen, best_value, best_candidate);
  for (int i = p_start; i < p_end; ++i) {
    update_verified_best(solver, solver.newpop[i], seen, best_value,
                         best_candidate);
    update_verified_best(solver, solver.pop[i], seen, best_value,
                         best_candidate);
    update_verified_best(solver, solver.ibest[i], seen, best_value,
                         best_candidate);
  }
#else
  seen.reserve(p_end - p_start);
  double best_value = 1e300;
  for (int i = p_start; i < p_end; ++i)
    update_verified_best(solver, solver.newpop[i], seen, best_value,
                         best_candidate);
#endif

  return best_value;
}

static void print_load_distribution(const char* name,
                                    const CEDLoadDistribution& load) {
  cout << name << " load: mean=" << load.mean
       << ", sd=" << load.standard_deviation << ", min=" << load.minimum
       << ", median=" << load.median << ", p95=" << load.percentile95
       << ", max=" << load.maximum << ", Jain=" << load.jain_index
       << ", total=" << load.total_count << ", used=" << load.used_count
       << ", active-Jain=" << load.active_jain_index
       << endl;
}

static void print_detailed_metrics(int rank, double local_objective,
                                   const CEDDetailedMetrics& local_metrics) {
  CEDDetailedMetrics metrics = local_metrics;
#ifdef USE_MPI
  struct { double value; int rank; } local_pair{local_objective, rank},
                                      global_pair{local_objective, rank};
  MPI_Allreduce(&local_pair, &global_pair, 1, MPI_DOUBLE_INT, MPI_MINLOC,
                MPI_COMM_WORLD);
  MPI_Bcast(&metrics, static_cast<int>(sizeof(metrics)), MPI_BYTE,
            global_pair.rank, MPI_COMM_WORLD);
#endif
  if (rank != 0)
    return;
  cout << "Raw energy = " << metrics.energy << endl;
  cout << "Makespan = " << metrics.makespan << endl;
  cout << "Mean manufacturing-operation queue wait = "
       << metrics.average_operation_queue_wait << endl;
  cout << "Mean task communication time = "
       << metrics.average_task_communication_time << endl;
  cout << "Total transportation time = "
       << metrics.total_transportation_time << endl;
  print_load_distribution("Cloud", metrics.cloud_load);
  print_load_distribution("Edge", metrics.edge_load);
  print_load_distribution("Industrial device", metrics.device_load);
  print_load_distribution("Factory busy time", metrics.factory_busy_time);
  print_load_distribution("Factory idle time", metrics.factory_idle_time);
  print_load_distribution("All-resource busy time",
                          metrics.all_resource_busy_time);
  print_load_distribution("All-resource idle time",
                          metrics.all_resource_idle_time);
  print_load_distribution("Cloud busy time", metrics.cloud_busy_time);
  print_load_distribution("Cloud idle time", metrics.cloud_idle_time);
  print_load_distribution("Edge busy time", metrics.edge_busy_time);
  print_load_distribution("Edge idle time", metrics.edge_idle_time);
  print_load_distribution("Industrial-device busy time",
                          metrics.device_busy_time);
  print_load_distribution("Industrial-device idle time",
                          metrics.device_idle_time);
}

static void run_generation(MultiMet& solver, int generation, unsigned base_seed,
                           int rank, int mpi_size, int p_start, int p_end,
                           double& current_best, double& current_average) {
  unsigned generation_seed = base_seed + generation * 104729;
#ifdef USE_MPI
  generation_seed += static_cast<unsigned>(rank) * 130363U;
#endif
  srand(generation_seed);
  const bool global_search_enabled = configured_global_search_enabled();
  if (global_search_enabled)
    run_global_search(solver, generation, p_start, p_end, rank);
#if defined(USE_MPI) && LOCAL_ARCHIVE_ENABLED
  seed_local_working_slot_from_archive(solver, p_start, p_end);
#endif
#ifdef USE_MPI
#if ROTATING_TRUE_AUDIT_INTERVAL > 0
  if ((generation + 1) % ROTATING_TRUE_AUDIT_INTERVAL == 0) {
    const int audit_round =
        (generation + 1) / ROTATING_TRUE_AUDIT_INTERVAL - 1;
    if (rank == audit_round % std::max(1, mpi_size)) {
      CED_ForceNextProxyTrueEvaluation();
#if defined(AUDIT_STRUCTURED_MEME) && AUDIT_STRUCTURED_MEME &&                 \
    !TRI_POLICY_RANDOM_ABLATION
      solver.forced_meme_action = 0;
#endif
    }
  }
#endif
#endif
#if STRICT_SINGLE_NO_MEMORY
  // The strict single-individual ablation has no cross-generation personal or
  // rank-local best memory. Keep the legacy storage required by the unchanged
  // Meme interface synchronized with the sole current individual.
  if (p_start < p_end) {
    const size_t bytes = sizeof(double) * static_cast<size_t>(solver.Nvar);
    std::memcpy(solver.ibest[0], solver.newpop[0], bytes);
    std::memcpy(solver.gbest, solver.newpop[0], bytes);
    solver.ibest_fit[0] = solver.newpop_fit[0];
    solver.gbest_fit = solver.newpop_fit[0];
    solver.meme_last_best = solver.newpop_fit[0];
  }
#endif
  run_meme_search(solver, generation, p_start, p_end);
  const bool full_population_commit = global_search_enabled;
  if (full_population_commit) {
#ifdef USE_MPI
    solver.pop_update(p_start, p_end);
#else
    solver.pop_update(0, solver.Popsize);
#endif
    solver.worst_and_best();
    current_best = solver.pop_fit[solver.cur_best];
    current_average = VERBOSE_OUTPUT ? solver.average_fit() : current_best;
    solver.Elist();
    solver.worst_and_best();
  } else {
#ifdef USE_MPI
    solver.meme_state_update(p_start, p_end);
#else
    solver.meme_state_update(0, solver.Popsize);
#endif
    solver.worst_and_best();
    current_best = solver.gbest_fit;
    current_average = VERBOSE_OUTPUT ? solver.average_fit() : current_best;
  }

#ifdef USE_MPI
  if (sync_before_next_generation(generation, MIGRATION_INTERVAL)) {
    migrate_best_solution(solver, rank, mpi_size, p_start);
    solver.worst_and_best();
    current_best = solver.pop_fit[solver.cur_best];
    current_average = solver.average_fit();
  }
#else
  (void)rank;
  (void)mpi_size;
#endif
}

int main() {
  int generation = 0;

#ifdef USE_MPI
  MPI_Init(NULL, NULL);
  int rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#if LOCAL_ARCHIVE_ENABLED
  const int solver_population_size = TNUM >= 1000000 ? 1 : POPSIZE;
#else
  const int solver_population_size = 1;
#endif
  int p_start = 0, p_end = solver_population_size;
  if (solver_population_size > 1)
    rank_range(rank, mpi_size, solver_population_size, p_start, p_end);
#else
  int rank = 0;
  int mpi_size = 1;
#if SERIAL_POPULATION_ABLATION
  const int solver_population_size = POPSIZE;
#else
  const int solver_population_size = TNUM >= 1000000 ? 1 : POPSIZE;
#endif
  int p_start = 0, p_end = solver_population_size;
#endif

  unsigned base_seed = make_seed(rank);
  srand(base_seed);
#if SURROGATE_ENABLED
  FF objective_func = CED_Schedule_ParallelProxy;
#else
  FF objective_func = CED_Schedule;
#endif
  MultiMet solver(solver_population_size,
                  TNUM * 2 + TNUM * MOPT_NUM * 2, 0, 1, CNUM, ENUM, DNUM,
                  TNUM, TNUM, MOPT_NUM, objective_func);
  solver.Initial();
#if TASK_PRIOR_PROXY_DESIGN
  apply_task_prior_proxy_design(solver);
#endif
#if DTGP_TASK_PRIOR
  apply_task_priors(solver, rank);
#endif
#ifdef USE_MPI
  if (solver_population_size > 1)
    sync_initial_population(solver);
#endif
#ifdef CONVERGENCE_TRACE
  {
    double initial_verified = CED_ProxyBestTrueValue();
#ifdef USE_MPI
    double global_initial_verified = initial_verified;
    MPI_Allreduce(&initial_verified, &global_initial_verified, 1, MPI_DOUBLE,
                  MPI_MIN, MPI_COMM_WORLD);
    initial_verified = global_initial_verified;
#endif
    if (rank == 0)
      cout << "CONVERGENCE,0," << initial_verified << endl;
  }
#endif
#ifdef EVOLUTION_IMPROVEMENT_AUDIT
  double initial_verified_best =
      verify_final_candidates_true(solver, p_start, p_end);
#ifdef USE_MPI
  double global_initial_verified_best = initial_verified_best;
  MPI_Allreduce(&initial_verified_best, &global_initial_verified_best, 1,
                MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  initial_verified_best = global_initial_verified_best;
#endif
#endif
  const int prewarm_generations =
#if STRICT_SINGLE_NO_MEMORY
      0;
#elif !SURROGATE_ENABLED
      0;
#elif TASK_PRIOR_PROXY_DESIGN
      0;
#else
      proxy_prewarm_generations(solver.Popsize);
#endif
  for (int warmup = 0; warmup < prewarm_generations; warmup++) {
    double warmup_best = 0.0;
    double warmup_average = 0.0;
    run_generation(solver, warmup, base_seed, rank, mpi_size, p_start, p_end,
                   warmup_best, warmup_average);
  }
#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  double start_time = current_time();

  while (generation < MAXGEN) {
    double current_best = 0.0;
    double current_average = 0.0;
    run_generation(solver, generation, base_seed, rank, mpi_size, p_start,
                   p_end, current_best, current_average);

    generation++;
#ifdef CONVERGENCE_TRACE
    {
      double local_verified = CED_ProxyBestTrueValue();
#ifdef USE_MPI
      double global_verified = local_verified;
      MPI_Allreduce(&local_verified, &global_verified, 1, MPI_DOUBLE, MPI_MIN,
                    MPI_COMM_WORLD);
      local_verified = global_verified;
#endif
      if (rank == 0)
        cout << "CONVERGENCE," << generation << ',' << local_verified << endl;
    }
#endif
    print_progress(rank, generation, current_best, current_average);
  }

  double elapsed_time = current_time() - start_time;
#ifdef USE_MPI
  double max_elapsed_time = elapsed_time;
  MPI_Reduce(&elapsed_time, &max_elapsed_time, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  if (rank == 0)
    elapsed_time = max_elapsed_time;
#endif
  vector<double> verified_candidate;
  double verified_best = verify_final_candidates_true(
      solver, p_start, p_end, &verified_candidate);
  if (!verified_candidate.empty())
    true_schedule_value(solver, verified_candidate.data());
  CEDDetailedMetrics verified_metrics = CED_LastDetailedMetrics();
#ifdef USE_MPI
  double global_verified_best = verified_best;
  MPI_Allreduce(&verified_best, &global_verified_best, 1, MPI_DOUBLE, MPI_MIN,
                MPI_COMM_WORLD);
  verified_best = global_verified_best;
#endif
  print_summary(rank, generation, verified_best, elapsed_time);
  print_detailed_metrics(rank, verified_metrics.objective, verified_metrics);
#ifdef EVOLUTION_IMPROVEMENT_AUDIT
  if (rank == 0) {
    const double absolute_improvement = initial_verified_best - verified_best;
    const double relative_improvement =
        absolute_improvement / (std::fabs(initial_verified_best) + 1e-12);
    std::cout << "Initial verified solution = " << initial_verified_best
              << std::endl;
    std::cout << "Evolutionary improvement = " << absolute_improvement
              << std::endl;
    std::cout << "Relative improvement = " << 100.0 * relative_improvement
              << "%" << std::endl;
  }
#endif
#ifdef POLICY_ABLATION_REPORT
  long long local_proxy_actions[2] = {0, 0};
  for (int state = 0; state < MultiMet::HyperStateCount; ++state) {
    for (int action = 0; action < MultiMet::MemeProxyPolicyActionCount;
         ++action) {
      local_proxy_actions[action] += solver.proxy_policy_trials[state][action];
    }
  }
  long long global_proxy_actions[2] = {0, 0};
  long long local_true_evaluations = CED_ProxyTrueEvaluationCount();
  long long global_true_evaluations = local_true_evaluations;
  long long local_cache_hits = CED_ProxyExactCacheHitCount();
  long long global_cache_hits = local_cache_hits;
#ifdef USE_MPI
  MPI_Reduce(local_proxy_actions, global_proxy_actions, 2, MPI_LONG_LONG,
             MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_true_evaluations, &global_true_evaluations, 1,
             MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_cache_hits, &global_cache_hits, 1, MPI_LONG_LONG, MPI_SUM,
             0, MPI_COMM_WORLD);
#else
  global_proxy_actions[0] = local_proxy_actions[0];
  global_proxy_actions[1] = local_proxy_actions[1];
#endif
  if (rank == 0)
    std::cout << "Proxy actions: reduce=" << global_proxy_actions[0]
              << ", conservative=" << global_proxy_actions[1]
              << ", true evaluations=" << global_true_evaluations
              << ", exact-cache hits=" << global_cache_hits
              << std::endl;
  long long local_configuration_trials[MultiMet::MemeConfigurationActionCount] = {};
  double local_configuration_weights[MultiMet::MemeConfigurationActionCount] = {};
  long long global_configuration_trials[MultiMet::MemeConfigurationActionCount] = {};
  double global_configuration_weights[MultiMet::MemeConfigurationActionCount] = {};
  for (int state = 0; state < MultiMet::HyperStateCount; ++state)
    for (int action = 0; action < MultiMet::MemeConfigurationActionCount;
         ++action) {
      local_configuration_trials[action] +=
          solver.configuration_policy_trials[state][action];
      local_configuration_weights[action] +=
          solver.meme_policy.configuration[state][action];
    }
#ifdef USE_MPI
  MPI_Reduce(local_configuration_trials, global_configuration_trials,
             MultiMet::MemeConfigurationActionCount, MPI_LONG_LONG, MPI_SUM,
             0, MPI_COMM_WORLD);
  MPI_Reduce(local_configuration_weights, global_configuration_weights,
             MultiMet::MemeConfigurationActionCount, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
#else
  std::copy(local_configuration_trials,
            local_configuration_trials + MultiMet::MemeConfigurationActionCount,
            global_configuration_trials);
  std::copy(local_configuration_weights,
            local_configuration_weights + MultiMet::MemeConfigurationActionCount,
            global_configuration_weights);
#endif
  if (rank == 0) {
    std::cout << "Configuration policy:";
    for (int action = 0; action < MultiMet::MemeConfigurationActionCount;
         ++action)
      std::cout << " a" << action << "=" << global_configuration_trials[action]
                << "/" << global_configuration_weights[action] /
                                   (mpi_size * MultiMet::HyperStateCount);
    std::cout << std::endl;
  }
  long long local_meme_trials[MultiMet::MemePolicyActionCount] = {};
  double local_meme_weights[MultiMet::MemePolicyActionCount] = {};
  long long global_meme_trials[MultiMet::MemePolicyActionCount] = {};
  double global_meme_weights[MultiMet::MemePolicyActionCount] = {};
  for (int state = 0; state < MultiMet::HyperStateCount; ++state)
    for (int action = 0; action < MultiMet::MemePolicyActionCount; ++action) {
      local_meme_trials[action] += solver.meme_policy_trials[state][action];
      local_meme_weights[action] += solver.meme_policy.meme[state][action];
    }
#ifdef USE_MPI
  MPI_Reduce(local_meme_trials, global_meme_trials,
             MultiMet::MemePolicyActionCount, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(local_meme_weights, global_meme_weights,
             MultiMet::MemePolicyActionCount, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);
#else
  std::copy(local_meme_trials,
            local_meme_trials + MultiMet::MemePolicyActionCount,
            global_meme_trials);
  std::copy(local_meme_weights,
            local_meme_weights + MultiMet::MemePolicyActionCount,
            global_meme_weights);
#endif
  if (rank == 0) {
    std::cout << "Meme policy:";
    for (int action = 0; action < MultiMet::MemePolicyActionCount; ++action)
      std::cout << " a" << action << "=" << global_meme_trials[action] << "/"
                << global_meme_weights[action] /
                       (mpi_size * MultiMet::HyperStateCount);
    std::cout << std::endl;
  }
  int global_verified_successes[MultiMet::MemePolicyActionCount] = {};
#ifdef USE_MPI
  MPI_Reduce(solver.meme_verified_successes, global_verified_successes,
             MultiMet::MemePolicyActionCount, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
#else
  std::copy(solver.meme_verified_successes,
            solver.meme_verified_successes + MultiMet::MemePolicyActionCount,
            global_verified_successes);
#endif
  if (rank == 0) {
    std::cout << "Verified Meme successes:";
    for (int action = 0; action < MultiMet::MemePolicyActionCount; ++action)
      std::cout << " a" << action << "=" << global_verified_successes[action];
    std::cout << std::endl;
  }
#endif
#ifdef USE_MPI
  MPI_Finalize();
#endif
  return 0;
}
