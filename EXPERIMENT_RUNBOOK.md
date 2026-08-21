# CED Scheduling Experiment Runbook

This document is the canonical guide for compiling, running, and reporting the current CED scheduling algorithm on the regional multi-factory industrial-Internet benchmark generated on 2026-08-07. Follow it exactly for all routine three-scale experiments unless the experiment explicitly studies a configuration change. The former fixed-resource data matrices are retained only as historical results and are not directly comparable with this benchmark.

## 1. Current algorithm

The formal algorithm consists of eight MPI subpopulations, an eight-entry local
archive in each process for the first three benchmark scales, eight Meme
operators, state-dependent Meme and proxy policy learning, the hybrid surrogate
model, fixed clockwise ring migration every ten generations, final verification
using the true scheduling objective, and persistent Robbins--Monro adaptation
for the Gaussian Meme. The memory-reduced million-task stress test retains its
original one-local-individual layout.

The formal algorithm does **not** use ADE or any other global evolutionary search. `GLOBAL_SEARCH_ENABLED` must remain `0`.

## 2. Canonical configuration

| Parameter | Required value |
|---|---:|
| `TNUM` | `1000`, `10000`, or `100000` |
| `MAXGEN` | `50` |
| `POPSIZE` | `8` |
| MPI process count | `8` |
| `MOPT_NUM` | `5` |
| `CNUM` | scale-dependent; see below |
| `ENUM` | scale-dependent; see below |
| `DNUM` | scale-dependent; see below |
| `MIGRATION_INTERVAL` | `10` |
| `MEME_SEARCH_MODE` | `0` |
| `GLOBAL_SEARCH_ENABLED` | `0` |
| `MPI_INDEPENDENT_SEED` | `0` |
| `BASE_SEED` | `20260616` |
| `VERBOSE_OUTPUT` | `0` |
| Compiler optimization | `-O2` |
| RBF kernel coefficient | `gamma = 24` |

The benchmark scales one factory-level resource unit with every 1,000 computational tasks and 1,000 manufacturing jobs:

| `TNUM` | Factories | `CNUM` | `ENUM` | `DNUM` |
|---:|---:|---:|---:|---:|
| 1,000 | 1 | 16 | 4 | 64 |
| 10,000 | 10 | 160 | 40 | 640 |
| 100,000 | 100 | 1,600 | 400 | 6,400 |

`CNUM` denotes logical cloud computing nodes rather than physical cloud data centers. Each factory occupies a synthetic 180 m by 80 m footprint, contains four on-premises edge nodes and 64 schedulable industrial devices, and is separated from neighboring factory centers by 2 km. The 180 m by 80 m footprint follows the example industrial facility in the 5G-ACIA industrial traffic model. The resource ratios and workload topology are explicit synthetic benchmark assumptions, not claimed empirical averages.

Do not define `MPI_INDEPENDENT_SEED=1`. Do not set `CED_SEED`. Do not enable `CED_GLOBAL_SEARCH_ENABLED`. These changes produce a different experiment and must not be compared directly with the canonical baseline.

## 3. Data files

| `TNUM` | Data file |
|---:|---|
| 1,000 | `/Users/lailiyuanjun/Desktop/data_generator/datamatrix_1000` |
| 10,000 | `/Users/lailiyuanjun/Desktop/data_generator/datamatrix_10000` |
| 100,000 | `/Users/lailiyuanjun/Desktop/data_generator/datamatrix_100000` |

The reproducible generator is `/Users/lailiyuanjun/Desktop/data_generator/data_generator/main_industrial_benchmark.cpp`, uses seed `20260807`, and writes a `.meta` sidecar for every matrix. Each manufacturing job has five operations, each operation has exactly three eligible industrial devices, and every computational task can use the four edge nodes in its factory. Computational dependencies are generated independently without switch fall-through. The computation--manufacturing coupling probabilities are 0.30 for no coupling, 0.25 for start coupling, 0.25 for completion coupling, and 0.20 for joint coupling.

Do not use `TNUM=100` as one of the formal three scales.

## 4. Robbins--Monro Gaussian adaptation

The Gaussian Meme uses a persistent step size within each MPI subpopulation:

\[
\sigma_{r+1}=\sigma_r\exp\!\left[\eta_r\left(\mathbb I_r-0.2\right)\right],
\]

\[
\eta_r=0.1\left(\frac{10}{N_r+10}\right)^{0.6}.
\]

The implementation is located in `CED_schedule/MultimethodMeme.cpp`. The step size and cumulative application count are stored in `meme_gaussian_sigma` and `meme_gaussian_trials`. They persist across individuals and generations in one subpopulation and are reset only when the solver state is initialized.

The RBF surrogate uses the fixed kernel

\[
K(\boldsymbol\phi,\boldsymbol\phi_s)
=\exp\!\left(-24\|\boldsymbol\phi-\boldsymbol\phi_s\|_2^2\right).
\]

The coefficient is fixed for every scale and throughout the complete run; it is not adapted online.

## 5. Pre-run verification

Before compiling, verify the canonical defaults:

```bash
rg -n "MAXGEN|POPSIZE|BASE_SEED|MPI_INDEPENDENT_SEED|GLOBAL_SEARCH_ENABLED|MEME_SEARCH_MODE" \
  CED_schedule/Config.h CED_schedule/main.cpp
```

Expected critical values:

```text
MAXGEN=50
POPSIZE=8
BASE_SEED=20260616
MPI_INDEPENDENT_SEED=0
GLOBAL_SEARCH_ENABLED=0
MEME_SEARCH_MODE=0
```

Check that the shell has no experiment overrides:

```bash
env | rg '^CED_' || true
```

## 6. Canonical compilation

Run from the repository root:

```bash
for scale in 1000 10000 100000; do
  .local/mpich/bin/mpic++ \
    -std=c++17 -O2 -DUSE_MPI \
    -DTNUM=$scale \
    -DMAXGEN=50 \
    -DPOPSIZE=8 \
    -DVERBOSE_OUTPUT=0 \
    CED_schedule/main.cpp \
    CED_schedule/Multimethod.cpp \
    CED_schedule/MultimethodMeme.cpp \
    CED_schedule/Problems.cpp \
    -o build/canonical_rm_t${scale}_g50 || exit 1
done
```

Do not add:

```text
-DMPI_INDEPENDENT_SEED=1
-DGLOBAL_SEARCH_ENABLED=1
-DGLOBAL_SEARCH_ALG=...
-DMAXGEN=500
```

## 7. Canonical execution

Run every scale twice using the same executable:

```bash
for scale in 1000 10000 100000; do
  for run in 1 2; do
    echo "TNUM=$scale RUN=$run"
    env \
      -u CED_SEED \
      -u CED_GLOBAL_SEARCH_ENABLED \
      -u CED_GLOBAL_SEARCH_ALG \
      -u CED_MEME_SEARCH_MODE \
      .local/mpich/bin/mpirun -np 8 \
      build/canonical_rm_t${scale}_g50
  done
done
```

Do not prefix the command with `CED_SEED=...`. Do not set a global-search environment variable.

## 8. Reporting convention

Use the program output:

```text
The best solution = ...
Time = ... s
```

Report `The best solution` as solution quality. It is recomputed by the true scheduling objective and reduced across all eight MPI processes.

Report the program's `Time` as algorithm search time. It covers the formal 50-generation loop and excludes data loading, initialization, proxy prewarming, final verification, and MPI startup/shutdown.

Do not substitute `/usr/bin/time` wall-clock output for algorithm time in the main table. End-to-end wall-clock time may be reported separately only when clearly labeled.

## 9. Current regional industrial benchmark results

Verified results for the finalized resource configuration generated with seed `20260807`, the workload/capacity normalization, the 10 m near-field wireless-distance floor, and algorithm seed `20260616` are:

| `TNUM` | Run 1 | Run 2 | Mean search time |
|---:|---:|---:|---:|
| 1,000 | `94.7984 / 0.003265 s` | `94.7984 / 0.003065 s` | `0.003165 s` |
| 10,000 | `154.507 / 0.016541 s` | `154.507 / 0.019547 s` | `0.018044 s` |
| 100,000 | `160.048 / 0.220699 s` | `160.048 / 0.207424 s` | `0.214062 s` |

The time denominator is the maximum of four workload-derived references: manufacturing workload per industrial device, the longest manufacturing job chain, computational workload per aggregate cloud-edge capacity, and the computational-task critical path. The energy denominator scales the time reference by the number of active cloud-edge nodes and adds the communication-energy reference. This makes the objective dimensionless and comparable in interpretation across the three proportionally scaled instances. The manufacturing-device decoder stores and retrieves assignments with the same operation index; this prevents artificial cross-factory transport. Wireless propagation uses a 10 m near-field reference-distance floor to prevent sub-meter random geometry from dominating interference.

The feasible-anchor test uses the absence sentinel `best_true>=1e299`; it must
not compare a normalized objective with `1.0`, because valid objectives in
this benchmark exceed one.

## 10. Policy-learned associated-block search (2026-08-10)

The current search uses no task-prior initialization
(`TASK_PRIOR_PROXY_DESIGN=0`) and no fixed policy preference. All policy
weights and selection counts start symmetrically. When the structured Meme is
selected, its configuration policy chooses among the legacy transformation
and three task-aware greedy transformations; its scope policy independently
chooses the number of complete computation-task/associated-operation blocks.
The scope candidates are `{512,640,768,896,1000,1000,1000,1000}` for the
single-factory instance and `{112,144,176,208,240,272,304,336}` blocks per
factory otherwise.

The learned action exploration rate is
`min(1,log(A)/max(1,N))`; exploitation remains a linear-weight roulette. The
proxy policy may request an exact audit only when the NN, binary-classifier,
and RBF estimators disagree beyond the online mean proxy error and the
error-calibrated candidate band can contain an improvement. Once policy
evidence exists, this extra audit is restricted to Meme actions within the
current exploration-dependent top-weight band. No state inherits weights or
counts from another state (there is no policy warm start).

For the memory-reduced million-task solver, each subpopulation contains one
active solution. Its action policy therefore remains uniformly exploratory until the
number of locally verified successes exceeds
`ceil(sqrt(A*log(A)))`; subsequent choices use the same learned policy as the
other scales. This evidence gate prevents a one-sample state from exploiting
spurious early preferences. Exact-value reuse stores two 64-bit candidate
hashes and the objective value instead of retaining up to 24 complete
high-dimensional solutions.

The latest fair Policy/Random comparison uses seed `20260616`, eight MPI
processes, 50 generations, `-O2`, identical data, and exact final verification:

| `TNUM` | Policy-8Meme objective / time | Random-8Meme objective / time |
|---:|---:|---:|
| 1,000 | `63.682 / 0.014132 s` | `70.9231 / 0.012458 s` |
| 10,000 | `129.145 / 0.121253 s` | `131.372 / 0.135027 s` |
| 100,000 | `131.376 / 1.31098 s` | `134.174 / 1.72686 s` |
| 1,000,000 | `130.660 / 20.6955 s` | `130.660 / 19.4417 s` |

Policy-8Meme is better at the first three scales and matches Random-8Meme at
one million tasks. Million-scale timings fluctuate from approximately 18 to
21 seconds under the current machine memory pressure, so those two timings
must be treated as equivalent unless independent repetitions show otherwise.

## 11. Historical fixed-resource baseline (not directly comparable)

Accepted results before the persistent Robbins--Monro update:

| `TNUM` | Candidate result | Repeated runtime |
|---:|---:|---:|
| 1,000 | `7.21417 / 0.004132 s` | `0.004490 s` |
| 10,000 | `14.5189 / 0.038037 s` | `0.041276 s` |
| 100,000 | `88.9373 / 0.456201 s` | `0.474812 s` |

Verified persistent Robbins--Monro results with the former `gamma = 36` setting:

| `TNUM` | Run 1 | Run 2 |
|---:|---:|---:|
| 1,000 | `7.21603 / 0.003299 s` | `7.21603 / 0.002732 s` |
| 10,000 | `14.5166 / 0.023848 s` | `14.5166 / 0.018138 s` |
| 100,000 | `88.9355 / 0.253093 s` | `88.9355 / 0.340217 s` |

The two repetitions verify deterministic solution quality under the fixed default seed. They are runtime repetitions, not independent statistical trials.

Verified results after fixing the RBF coefficient to `gamma = 24`:

| `TNUM` | Run 1 | Run 2 |
|---:|---:|---:|
| 1,000 | `7.21603 / 0.002874 s` | `7.21603 / 0.003218 s` |
| 10,000 | `14.5166 / 0.019399 s` | `14.5166 / 0.025966 s` |
| 100,000 | `88.9355 / 0.262066 s` | `88.9355 / 0.256929 s` |

Changing `gamma` from 36 to 24 did not change the verified objective value at any of the three scales.

## 12. Mandatory checklist

- [ ] `TNUM` is 1,000, 10,000, or 100,000.
- [ ] `MAXGEN=50`.
- [ ] `POPSIZE=8`.
- [ ] Compilation uses `-DUSE_MPI` and `-O2`.
- [ ] Execution uses `mpirun -np 8`.
- [ ] ADE and all other global searches are disabled.
- [ ] `MPI_INDEPENDENT_SEED=0`.
- [ ] No `CED_SEED` override is present.
- [ ] No other `CED_*` override is present.
- [ ] The correct data file is selected.
- [ ] `CNUM`, `ENUM`, and `DNUM` match the scale-dependent benchmark table.
- [ ] Quality comes from `The best solution`.
- [ ] Runtime comes from the program's `Time`.
- [ ] Both repetitions complete normally.
- [ ] Any changed setting is labeled as an ablation and is not mixed with the canonical baseline.

## 13. Optional million-task stress test

`TNUM=1000000` is an extended stress test and is not part of the canonical
three-scale baseline. It uses 1,000 factories, 16,000 cloud nodes, 4,000 edge
nodes, and 64,000 industrial devices. The compact input is
`/Users/lailiyuanjun/Desktop/data_generator/datamatrix_1000000_compact`.
It stores node coordinates instead of the full edge--device and device--device
distance matrices; distances are reconstructed on demand with the same
`DISTANCE_SCALE` quantization.

To fit the 24 GiB test machine, each of the eight MPI processes stores one local
individual and omits arrays belonging exclusively to disabled historical global
search algorithms. Rank-specific seeds are deterministic derivatives of
`BASE_SEED`. The original three scales retain their canonical population,
seeding, data format, and allocation paths.

Historical pre-associated-block stress-test result on 2026-08-08 (superseded
by Section 10):

| `TNUM` | MPI processes | Generations | Best objective | Search time |
|---:|---:|---:|---:|---:|
| 1,000,000 | 8 | 50 | `184.33` | `0.795645 s`, repeat `1.068600 s` |

The million-task run constructs the eight-point surrogate design before the
reported search timer. The reduce-check action retains the periodic audit,
whereas the conservative-check action additionally applies the original error,
optimism, and ranking-reliability conditions. The final reported solution is
always recomputed with the exact scheduling objective after the timer.
