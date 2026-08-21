# UCP-PSE

Source code, four-scale datasets, baseline adaptations, ablation settings and
reproducibility material for *Policy-Guided Generative Coding Evolution for
Million-Scale Cloud-Edge-Device Collaborative Task Scheduling*.

## Algorithm overview

UCP-PSE is a Uniform Control Policy-Guided Parallel Sparse Evolution framework
for large cloud-edge-device collaborative scheduling. A scheduling solution is
encoded as random keys covering manufacturing-device assignment, operation
priority, cloud/edge placement and compute-resource assignment.

The framework contains four main components:

1. **Sparse generative-coding operator.** Only selected variables or complete
   task-associated blocks are modified. Directional evolution, local
   perturbation and problem-guided transformations are combined without a
   full-dimensional update at every step.
2. **Lightweight hybrid surrogate.** A 24-feature representation feeds nearest
   neighbor, binary-classifier and RBF estimators plus two-stage residual
   correction. Promising or uncertain candidates are audited with the exact
   scheduling objective; every final reported solution is exact.
3. **Uniform control policy.** State-dependent policies learn sampling,
   perturbation, guidance and surrogate actions from cost-aware improvement
   rewards. No offline training is required.
4. **Parallel sparse evolution.** Eight MPI subpopulations search independently,
   retain local verified archives and exchange their best solutions through a
   ring every ten generations.

The public benchmark contains `10^3`, `10^4`, `10^5` and `10^6`
computational/manufacturing task pairs. The million-task case uses compact
geometry and one active individual per MPI rank to control memory use.

## Repository contents

| Path | Purpose |
|---|---|
| `CED_schedule/` | UCP-PSE, policy, surrogate and exact scheduling objective |
| `standalone_ced/` | Seven standalone matched-budget baseline adaptations |
| `data/` | Metadata, checksums and power data; matrices are Release assets |
| `tools/` | Dataset generators, downloader, experiment and summary scripts |
| `docs/` | Function guides, baseline provenance, ablations and dataset manual |
| `EXPERIMENT_RUNBOOK.md` | Canonical configuration and reporting contract |

All published C/C++ and shell sources contain paragraph-level comments. The
nine comparison algorithms are matched-budget CED adaptations; they are not
claimed to be exact reproductions of the source papers' native benchmarks.

## Requirements

- A C++17 compiler
- An MPI implementation providing `mpic++` and `mpirun`
- Eight available MPI processes
- `curl`, `gzip` and `shasum` for dataset installation
- Approximately 1 GB free disk space for downloaded and expanded datasets

The paper configuration uses `-O2`, `MAXGEN=50`, `POPSIZE=8`, eight MPI
processes and the seed rules in `EXPERIMENT_RUNBOOK.md`.

## 1. Clone and download all datasets

The expanded `10^5` and `10^6` matrices exceed normal Git file limits, so all
four matrices are distributed through the public
[`datasets-v1` Release](https://github.com/yunglynn/UCP-PSE/releases/tag/datasets-v1).

```bash
git clone https://github.com/yunglynn/UCP-PSE.git
cd UCP-PSE
tools/download_datasets.sh
```

The downloader resumes interrupted transfers, verifies the compressed assets,
extracts them into `data/`, and verifies the expanded files again. Before any
experiment, the following command must succeed:

```bash
(cd data && shasum -a 256 -c SHA256SUMS)
```

## 2. Build UCP-PSE

Build the three formal baseline scales and the optional million-task stress
test from the repository root:

```bash
mkdir -p build
for scale in 1000 10000 100000 1000000; do
  mpic++ -std=c++17 -O2 -DUSE_MPI \
    -DTNUM=$scale -DMAXGEN=50 -DPOPSIZE=8 -DVERBOSE_OUTPUT=0 \
    CED_schedule/main.cpp \
    CED_schedule/Multimethod.cpp \
    CED_schedule/MultimethodMeme.cpp \
    CED_schedule/Problems.cpp \
    -o build/ucp_pse_t${scale}
done
```

The default data paths are relative to the repository:

```text
data/datamatrix_1000
data/datamatrix_10000
data/datamatrix_100000
data/datamatrix_1000000_compact
```

To use a matrix elsewhere, add a compile-time override such as:

```bash
-DDATA_FILE_PATH='"/absolute/path/to/datamatrix_10000"'
```

## 3. Run UCP-PSE

The canonical baseline uses eight MPI ranks and no experiment environment
overrides:

```bash
env -u CED_SEED \
    -u CED_GLOBAL_SEARCH_ENABLED \
    -u CED_GLOBAL_SEARCH_ALG \
    -u CED_MEME_SEARCH_MODE \
    mpirun -np 8 build/ucp_pse_t1000
```

Replace `1000` with `10000` or `100000` for the other formal scales.
`1000000` is the optional compact stress test and must be reported separately
from the canonical three-scale baseline.

Critical output fields are:

```text
The best solution = ...
Time = ... s
```

`The best solution` is recomputed with the exact scheduling objective and
reduced across MPI ranks. `Time` is algorithm search-loop time. Do not replace
it with shell wall-clock time in the main result table.

## 4. Run repeated paper comparisons

The ten-run comparison of UCP-PSE and the nine paper baselines is automated by:

```bash
tools/run_component_vs_nine_metrics_10x.sh
```

It uses the four scales and paired seeds `20260616` through `20260625`. Review
the script and `docs/BASELINES_GUIDE.md` before running: the experiment is much
larger than the single canonical check, and every method must use the same
budget, data, process count and exact final audit.

The nine baseline rows are MadDE, QPHH, FCA-G, SoEA-BBRL, HGA,
Bi-Population CDE, AMTSA, NL-SHADE-LBC and SLPSO-ARS. Seven have standalone
drivers under `standalone_ced/`; NL-SHADE-LBC and SLPSO-ARS use
`CED_schedule/main_recent_algorithms.cpp`.

## 5. Generate new datasets

The published matrices should be used for paper reproduction. New robustness
instances can be generated with:

```bash
c++ -std=c++17 -O2 tools/main_industrial_benchmark_dense.cpp \
  -o build/generate_dense
c++ -std=c++17 -O2 tools/main_industrial_benchmark_compact.cpp \
  -o build/generate_compact

build/generate_dense 1000 data/new_datamatrix_1000
build/generate_compact 1000000 data/new_datamatrix_1000000_compact
```

Newly generated matrices that do not match `data/SHA256SUMS` are new benchmark
instances and must not be mixed with the paper tables as if they were the
published data.

## Detailed documentation

- [`docs/INDEX.md`](docs/INDEX.md): complete paper-to-code index
- [`EXPERIMENT_RUNBOOK.md`](EXPERIMENT_RUNBOOK.md): protected canonical settings
- [`docs/UCP_PSE_GUIDE.md`](docs/UCP_PSE_GUIDE.md): UCP-PSE functions and flow
- [`docs/BASELINES_GUIDE.md`](docs/BASELINES_GUIDE.md): nine baseline sources and use
- [`docs/ABLATIONS_GUIDE.md`](docs/ABLATIONS_GUIDE.md): five paper ablations
- [`docs/DATASETS_GUIDE.md`](docs/DATASETS_GUIDE.md): download, formats,
  generation, validation and troubleshooting

For reproducible reporting, record the commit, compiler/MPI versions, complete
compile command, seed, process count, dataset checksum, stdout and timing
convention with every result.
