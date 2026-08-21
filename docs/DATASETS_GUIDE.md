# Four-scale dataset generation and detailed usage

## 1. Dataset identity

The paper uses one deterministic regional multi-factory IIoT benchmark at four
task scales. A task count `TNUM=N` means `N` computational tasks and `N`
manufacturing jobs. Every manufacturing job contains five ordered operations,
so the real-valued random-key solution dimension is `12*TNUM`.

| `TNUM` | Factories | Cloud nodes | Edge nodes | Devices | Operations | Stored format |
|---:|---:|---:|---:|---:|---:|---|
| 1,000 | 1 | 16 | 4 | 64 | 5,000 | dense |
| 10,000 | 10 | 160 | 40 | 640 | 50,000 | dense |
| 100,000 | 100 | 1,600 | 400 | 6,400 | 500,000 | dense |
| 1,000,000 | 1,000 | 16,000 | 4,000 | 64,000 | 5,000,000 | compact |

The first three matrices are the canonical dense files used by the paper.
The million-task case uses `CED_COMPACT_V1`, which stores coordinates instead
of quadratic distance matrices. Do not replace a dense paper matrix with a
new compact file and call it the same experiment without an equivalence test.

## 2. Download the published matrices

From the repository root:

```bash
tools/download_datasets.sh
```

The script downloads the four `.gz` assets from GitHub Release `datasets-v1`,
checks `data/ASSET_SHA256SUMS`, extracts them under `data/`, then checks the
uncompressed files against `data/SHA256SUMS`. Expected disk use after download
and extraction is about 1.0 GB.

Manual download is also possible:

```bash
base=https://github.com/yunglynn/UCP-PSE/releases/download/datasets-v1
curl -fLO "$base/datamatrix_1000.gz"
curl -fLO "$base/datamatrix_10000.gz"
curl -fLO "$base/datamatrix_100000.gz"
curl -fLO "$base/datamatrix_1000000_compact.gz"
```

Always validate both compressed and uncompressed checksums. A successful
download must produce these uncompressed hashes:

| File | SHA-256 |
|---|---|
| `datamatrix_1000` | `c1e0cb07395d44dabc3bedcfdfb927f176302d4a182942ca0d67d26451f60ef1` |
| `datamatrix_10000` | `ec76c4e862726f840eb44d17b6b6d01e21134a74384168907f1197161e237aba` |
| `datamatrix_100000` | `64a660f7fab4413be0ecb14325addb629fdd7c70c155abe6e5170d8f1ba69c4c` |
| `datamatrix_1000000_compact` | `e8a93f0bb4c1a72d0e1460b495e9c27a6dcf85ed790b5f9c8f28d75bc8cdad37` |

## 3. Synthetic benchmark assumptions

The generator seed is `20260807`. Each factory represents a synthetic 180 m
by 80 m industrial footprint; adjacent factory centers are separated by 2 km.
Each factory contains 16 logical cloud compute nodes, four on-premises edge
nodes and 64 schedulable industrial devices.

Every manufacturing operation has exactly three eligible devices. Every
computational task can use the four edge nodes in its own factory. Coupling is
sampled as 0.30 no coupling, 0.25 start coupling, 0.25 completion coupling and
0.20 joint coupling. These ratios and the topology are reproducible synthetic
assumptions, not empirical population averages.

## 4. Generate the dense canonical format

The annotated dense generator is
`tools/main_industrial_benchmark_dense.cpp`. Compile it once:

```bash
mkdir -p build data
c++ -std=c++17 -O2 tools/main_industrial_benchmark_dense.cpp \
  -o build/generate_dense
```

Generate the first three scales:

```bash
build/generate_dense 1000 data/datamatrix_1000
build/generate_dense 10000 data/datamatrix_10000
build/generate_dense 100000 data/datamatrix_100000
```

The dense representation explicitly materializes every edge-to-device and
device-to-device distance. Its space grows quadratically with devices; a dense
million-task matrix is approximately 53 GB and is deliberately not published.

## 5. Generate the compact format

The compact generator is `tools/main_industrial_benchmark_compact.cpp`:

```bash
c++ -std=c++17 -O2 tools/main_industrial_benchmark_compact.cpp \
  -o build/generate_compact
build/generate_compact 1000000 data/datamatrix_1000000_compact
```

`CED_COMPACT_V1` starts with edge/device coordinates. The loader reconstructs
distances on demand using the same `DISTANCE_SCALE` quantization and 10 m
near-field floor as the objective. The remainder stores processing times,
computational loads, communication volumes, precedence and coupling lists,
eligible device lists, eligible edge lists and resource properties.

Both generators require `TNUM` to be a positive multiple of 1,000 and write a
`.meta` sidecar. Preserve the sidecar with the matrix.

## 6. Use the data with UCP-PSE

After `tools/download_datasets.sh`, run from the repository root. The public
source defaults to:

```text
data/datamatrix_1000
data/datamatrix_10000
data/datamatrix_100000
data/datamatrix_1000000_compact
```

Compile a scale exactly as specified in `EXPERIMENT_RUNBOOK.md`. If the matrix
is stored elsewhere, override the compile-time path with a quoted macro:

```bash
-DDATA_FILE_PATH='"/absolute/path/to/datamatrix_10000"'
```

For standalone baselines use:

```bash
-DSTANDALONE_DATA_FILE_PATH='"/absolute/path/to/datamatrix_10000"'
```

The value of `TNUM` must match the selected file. `CNUM`, `ENUM` and `DNUM`
must match the scale table; otherwise the loader may reject the file or decode
incorrect resource indices.

## 7. Validation before experiments

Run:

```bash
cd data
shasum -a 256 -c SHA256SUMS
cd ..
```

Then verify:

- the `.meta` task, factory and resource counts match the build;
- `operations_per_job=5`;
- `eligible_devices_per_operation=3`;
- the solver prints the intended data path and finishes loading normally;
- the build uses the canonical seed, process count and compiler settings;
- final quality comes from `The best solution`, not a proxy prediction;
- runtime comes from the program's `Time`, not shell wall-clock time.

## 8. Common problems

`Failed to open data file` means the download was not extracted, the program
was launched outside the repository root, or the compile-time override is
wrong. Use an absolute override or run from the repository root.

`Invalid or incomplete data file` usually means `TNUM` does not match the
matrix, the file is truncated, or a compact file was passed to an older loader.
Re-run both SHA-256 checks before debugging the algorithm.

An out-of-memory failure with the dense million-task file is expected. Use
`datamatrix_1000000_compact`; do not lower task/resource counts and report it
as the paper's million-task instance.

Generated matrices that do not match the published hashes are new benchmark
instances. They may be useful for robustness tests, but must be labelled as
new data and must not be mixed into the paper's paired baseline tables.
