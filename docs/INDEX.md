# Reproducibility index

## Reading order

1. [`../EXPERIMENT_RUNBOOK.md`](../EXPERIMENT_RUNBOOK.md): canonical seeds,
   compiler flags, MPI layout, timing convention, and protected defaults.
2. [`UCP_PSE_GUIDE.md`](UCP_PSE_GUIDE.md): UCP-PSE functions, call flow,
   configuration, and paper-to-code mapping.
3. [`BASELINES_GUIDE.md`](BASELINES_GUIDE.md): nine baseline publications,
   implementation locations, functions, commands, and adaptation boundaries.
4. [`ABLATIONS_GUIDE.md`](ABLATIONS_GUIDE.md): the five paper ablations and
   exact switches needed to reproduce them.
5. [`DATASETS_GUIDE.md`](DATASETS_GUIDE.md): four-scale generator, file format,
   validation, and use.

## Source map

| Paper item | Implementation | Driver |
|---|---|---|
| UCP-PSE orchestration, archive, ring migration | `CED_schedule/main.cpp` | same |
| Sparse generative-coding operator and UCP | `CED_schedule/MultimethodMeme.cpp` | `main.cpp` |
| Hybrid surrogate and exact-evaluation gate | `CED_schedule/Problems.cpp` | `main.cpp` |
| Exact CED decoder/objective | `CED_schedule/Problems.cpp` | all solvers |
| Shared population and policy state | `CED_schedule/Multimethod.h`, `Population.h` | all UCP paths |
| MadDE | `standalone_ced/madde.cpp` | same |
| QPHH | `standalone_ced/qphh.cpp` | same |
| FCA-G | `standalone_ced/fca_g.cpp` | same |
| SoEA-BBRL | `standalone_ced/soea.cpp` | same |
| HGA | `standalone_ced/hga.cpp` | same |
| Bi-Population CDE | `standalone_ced/bipop_cde.cpp` | same |
| AMTSA | `standalone_ced/amtsa.cpp` | same |
| NL-SHADE-LBC | `CED_schedule/RecentSchedulingAlgorithms.cpp` | `main_recent_algorithms.cpp` |
| SLPSO-ARS | `CED_schedule/RecentSchedulingAlgorithms.cpp` | `main_recent_algorithms.cpp` |
| Compact four-scale generator | `tools/main_industrial_benchmark_compact.cpp` | same |
| Ten-run paper comparison | `tools/run_component_vs_nine_metrics_10x.sh` | shell |

## Result provenance

Only outputs produced under the settings documented by the runbook and the
paper-specific ten-seed scripts may be used in the paper tables. Old ZIP files,
`benchmark_results_four_scales.txt`, and dated exploratory notes are historical
artifacts and are deliberately omitted from the GitHub source archive.

The program reports `The best solution` after exact verification. `Time` is
search-loop time. `/usr/bin/time` is end-to-end wall time and must not replace
the paper's search-time metric.
