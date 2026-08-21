# Four-scale dataset generation and use

## Scales

| Tasks/jobs | Factories | Cloud nodes | Edge nodes | Devices | Format |
|---:|---:|---:|---:|---:|---|
| 1,000 | 1 | 16 | 4 | 64 | dense |
| 10,000 | 10 | 160 | 40 | 640 | dense |
| 100,000 | 100 | 1,600 | 400 | 6,400 | dense |
| 1,000,000 | 1,000 | 16,000 | 4,000 | 64,000 | compact |

Each job has five manufacturing operations. Each operation has exactly three
eligible devices. Every computational task can use the four edge nodes in its
factory. Factory footprints are 180 m by 80 m and centers are spaced 2 km.
Coupling probabilities are 0.30 none, 0.25 start, 0.25 completion, and 0.20
joint. These are reproducible synthetic benchmark assumptions.

## Generator

The versioned generator is `tools/main_industrial_benchmark_compact.cpp` and
uses base seed 20260807 plus the task count in its deterministic random engine.
It accepts:

```bash
generator TNUM OUTPUT_PATH
```

Build it with a C++17 compiler. Generate into a data directory outside Git:

```bash
c++ -std=c++17 -O2 tools/main_industrial_benchmark_compact.cpp -o build/generator
build/generator 1000 /path/to/datamatrix_1000_compact
build/generator 10000 /path/to/datamatrix_10000_compact
build/generator 100000 /path/to/datamatrix_100000_compact
build/generator 1000000 /path/to/datamatrix_1000000_compact
```

The generator writes a `.meta` sidecar. Preserve it and record SHA-256 hashes
for both files. The historical dense files for the first three scales are
listed in `EXPERIMENT_RUNBOOK.md`; use those for the canonical baseline unless
an explicitly labelled compact-format equivalence test has verified identical
decoded objectives.

## Compact format

`CED_COMPACT_V1` stores edge and device coordinates rather than quadratic
distance matrices. Distances are reconstructed on demand with the same
quantization and near-field floor as the dense objective. Remaining sections
store operation durations, computational loads, communication volumes,
precedence/coupling data, eligible resource lists, and resource properties.

## Validation checklist

- `.meta` task/resource counts match the scale table.
- `operations_per_job=5` and `eligible_devices_per_operation=3`.
- Generation completes without duplicate eligible devices.
- The loader accepts the file with the intended `TNUM` build.
- A fixed known solution has the same exact objective after a round trip.
- File and metadata SHA-256 values are recorded with the run.
- Dataset files are not committed if their size or license violates GitHub
  limits; publish hashes and a release asset or data-repository link instead.
