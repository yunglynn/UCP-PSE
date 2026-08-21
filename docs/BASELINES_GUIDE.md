# Nine baseline implementations

## Scientific scope

All rows use the same CED instance, random-key decoder, exact objective,
`NP=8`, 50 generations, eight MPI ranks, paired seeds, and exact final audit.
Their source operators are adapted to this common representation and budget.
Accordingly, report them as **matched-budget CED adaptations**, not as exact
reproductions of the source papers' native benchmarks.

## Publication and implementation index

| Method | Source publication / provenance | Local implementation |
|---|---|---|
| MadDE | Biswas et al., “MadDE: A novel ensemble of mutation strategies and control parameters for differential evolution,” IEEE CEC 2021, DOI `10.1109/CEC45853.2021.9504792`; author MATLAB snapshot retained locally | `standalone_ced/madde.cpp` |
| QPHH | “A Reinforcement Learning-Based Population Hyper-Heuristic for Energy-Efficient Cloud Workflow Scheduling Problem,” IEEE TSC 18(5), 2025, DOI `10.1109/TSC.2025.3589126`; author C++ workflow source retained locally | `standalone_ced/qphh.cpp` |
| FCA-G | Yu et al. (2025), the paper's cited FCA-G method; author MATLAB/Python repository retained under `third_party/reproduction_sources/FCA-G` | `standalone_ced/fca_g.cpp` |
| SoEA-BBRL | Jiang et al. (2025), the paper's cited SoEA-BBRL scheduling method; author archive retained under `third_party/reproduction_sources/Soft-Scheduling/SoEA` | `standalone_ced/soea.cpp` |
| HGA | Mendes et al. (2025), the paper's cited hybrid genetic algorithm; author repository retained under `third_party/reproduction_sources/HGA-HFS` | `standalone_ced/hga.cpp` |
| Bi-Population CDE | IEEE TEVC 28(6), 2024, DOI `10.1109/TEVC.2023.3325004`; paper-guided block-decomposition mapping | `standalone_ced/bipop_cde.cpp` |
| AMTSA | IEEE TEVC, DOI `10.1109/TEVC.2026.3659072`; paper-guided single-objective CED mapping because the author archive was unavailable | `standalone_ced/amtsa.cpp` |
| NL-SHADE-LBC | Stanovov et al., IEEE CEC 2022 top-ranked submission; official C++ snapshot retained locally | `CED_schedule/RecentSchedulingAlgorithms.cpp` |
| SLPSO-ARS | Jian et al., IEEE TEVC 25(4), 2021, DOI `10.1109/TEVC.2021.3065659`; equation-level implementation | `CED_schedule/RecentSchedulingAlgorithms.cpp` |

Where the draft citation does not yet expose a DOI in the supplied manuscript,
the table deliberately records the paper citation key and local primary
artifact instead of inventing metadata. Complete bibliographic entries should
be copied from the manuscript's final `.bib` before public release.

## Function-level guide

Each `standalone_ced/*.cpp` owns initialization, variation/adaptation,
selection, MPI best reduction, exact final evaluation, and reporting. Shared
infrastructure is intentionally limited to:

- `ced_problem.{h,cpp}`: instance loading and exact objective wrapper;
- `benchmark_config.h`: scale/resource constants;
- `mpi_support.h`: deterministic seeding, partitioning, reductions, timing,
  and common reporting.

Algorithm stages:

- `madde.cpp`: multi-strategy mutation, adaptive strategy probabilities,
  parameter memories, archive-assisted selection.
- `qphh.cpp`: Q-state construction, three low-level scheduling actions,
  reward/Q update, action-controlled neighborhood search.
- `fca_g.cpp`: sparse CED relation context, factor construction, grouped
  variation, greedy environmental selection.
- `soea.cpp`: solution evolution plus BBRL-style rule selection and reward.
- `hga.cpp`: genetic reproduction, hybrid local improvement, survivor update.
- `bipop_cde.cpp`: decision-block decomposition, local/global populations,
  current-to-best/rand mutation, diversity-triggered regeneration.
- `amtsa.cpp`: adaptive structural versus task-split mode; bottleneck-guided
  spatial/temporal changes.
- `RecentSchedulingAlgorithms::NLSHADELBStep`: SHADE memories, linear
  population-pressure approximation under fixed NP, archive mutation.
- `RecentSchedulingAlgorithms::SLPSOARSStep`: level-based learning and
  adaptive random sparse updates.

## Build and run

The authoritative all-method script is:

```bash
tools/run_component_vs_nine_metrics_10x.sh
```

It compiles every scale with `-O2`, executes paired seeds 20260616--20260625,
requires the exact metrics in each log, and summarizes only complete outputs.
For NL-SHADE-LBC and SLPSO-ARS, `CED_COMPARISON_ALG` selects the method through
`main_recent_algorithms.cpp`; the other seven have separate executables.

Never compare a baseline run using a different population, generation count,
seed set, objective path, or MPI count with the paper table without explicitly
labelling it as a separate ablation.
