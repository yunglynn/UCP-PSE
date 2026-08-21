# UCP-PSE implementation guide

## Execution flow

`CED_schedule/main.cpp` is the executable entry. It validates the scale,
initializes MPI and the deterministic random stream, loads the instance,
constructs one solver per MPI rank, initializes the local archive, optionally
prewarms the surrogate, runs 50 generations, performs ring migration every ten
generations, and finally recomputes candidate quality with the exact objective.

The million-task path uses one active individual per rank and compact geometry.
This is a documented stress-test layout, not a silent replacement for the
three-scale population layout.

## Core files and functions

### `main.cpp`

- scale-specific matrix selection in `main`: chooses the dense or compact file.
- constructor/initialization block in `main`: creates the verified local state.
- `seed_local_working_slot_from_archive`: chooses the current archive parent.
- `migrate_best_solution`: sends the best verified member around the ring.
- the generation loop in `main`: executes sparse search, policy update, and
  migration.
- `verify_final_candidates_true`: exact final audit and MPI minimum reduction.
- `main`: owns setup, prewarm, timed loop, verification, and reporting.

### `MultimethodMeme.cpp`

The file is organized as a pipeline: state and feature helpers; four sampling
rules; directional references; five perturbations; four problem-guidance
rules; action selection; candidate construction; reward calculation; and
policy update.

- `MemePolicyState`: maps normalized progress to Early/Middle/Late.
- `SelectPolicy*Action`: applies state-local exploration plus weight roulette.
- `meme_selection`: dispatches the selected sparse rule.
- `RunMemeSearchStrategy`: generates, evaluates, accepts, and learns from one
  generation.
- `meme_state_update`: updates shared best/history state after acceptance.
- `meme_gaussian_sigma` / `meme_gaussian_trials`: persistent Robbins--Monro
  state for adaptive Gaussian perturbation.

The task-associated block operator modifies the computational task together
with its manufacturing-operation variables. The scope candidates are defined
per factory, preventing a scale increase from silently turning sparse search
into a full-vector update.

### `Problems.cpp`

- `readData`: loads dense or `CED_COMPACT_V1` input and validates dimensions.
- `CED_Schedule`: exact random-key decode and scheduling objective.
- `CED_Schedule_ParallelProxy`: hybrid-surrogate evaluation path.
- `CED_ProxyBestTrueValue`: best exact value known by the proxy archive.
- `CED_ProxyTrueEvaluationCount`: audit counter used by matched-budget tests.
- `CED_LastDetailedMetrics`: energy, makespan, communication, transport,
  waiting-time, and resource-activation metrics for the last exact evaluation.

The surrogate forms 24 features (six bins for each of four decision blocks),
combines nearest-neighbor, binary-classifier, and RBF estimates, then applies a
two-stage residual correction. The RBF coefficient is fixed at 24.

## Configuration

Canonical compile-time defaults are in `Config.h` and at the top of `main.cpp`.
The protected values are `MAXGEN=50`, `POPSIZE=8`, `MOPT_NUM=5`,
`BASE_SEED=20260616`, `MPI_INDEPENDENT_SEED=0`,
`GLOBAL_SEARCH_ENABLED=0`, `MEME_SEARCH_MODE=0`, `MIGRATION_INTERVAL=10`, and
`-O2`. Use exactly eight MPI processes.

Do not set `CED_SEED` for the deterministic canonical baseline. The ten-run
paper experiment intentionally uses common paired seeds 20260616--20260625;
that is a labelled statistical experiment, not the single-seed baseline.

## Canonical use

Use the build and run commands in `EXPERIMENT_RUNBOOK.md`. Before every run,
inspect `env | rg '^CED_'`; no hidden override may leak into a baseline.

## Paper mapping

| Paper component | Code region |
|---|---|
| Random-key encoding and exact model | `Problems.cpp` |
| Parallel subpopulations and archive | `main.cpp` archive helpers |
| Ring migration | `main.cpp` migration helpers |
| Sparse mask and task blocks | `MultimethodMeme.cpp` sampling helpers |
| Directional evolution | `MultimethodMeme.cpp` reference construction |
| Local perturbation | `MultimethodMeme.cpp` Meme operators |
| Problem-prior guidance | `MultimethodMeme.cpp` greedy configuration helpers |
| Hybrid surrogate | `Problems.cpp` proxy state and estimators |
| Four-layer UCP | `MultimethodMeme.cpp` action selection and weight updates |
