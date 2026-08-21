# UCP-PSE reproducibility package

This repository contains the implementation and reproducibility material for
*Policy-Guided Generative Coding Evolution for Million-Scale Cloud-Edge-Device
Collaborative Task Scheduling*.

Start with [`docs/INDEX.md`](docs/INDEX.md). It maps every paper component,
baseline, ablation, data scale, build entry point, and output convention to the
corresponding source file. The canonical experimental contract is
[`EXPERIMENT_RUNBOOK.md`](EXPERIMENT_RUNBOOK.md); it takes precedence over old
notes and historical result files.

## Quick start: download the required data first

The large matrices are attached to the public
[`datasets-v1` GitHub Release](https://github.com/yunglynn/UCP-PSE/releases/tag/datasets-v1)
rather than stored in Git history. A plain clone therefore contains metadata
and checksums but not the four expanded matrices. Prepare them with one command:

```bash
git clone https://github.com/yunglynn/UCP-PSE.git
cd UCP-PSE
tools/download_datasets.sh
```

The script uses `curl`, resumes interrupted downloads, verifies every `.gz`
asset, extracts it under `data/`, and verifies the expanded matrix again. A
successful run ends with `All four UCP-PSE datasets are ready`.

Do not run an experiment if `shasum -a 256 -c data/SHA256SUMS` fails. See
[`docs/DATASETS_GUIDE.md`](docs/DATASETS_GUIDE.md) for manual download,
generation, format, path override, disk-space and troubleshooting details.

Important scope labels:

- `CED_schedule/` contains UCP-PSE and the shared exact scheduling objective.
- `standalone_ced/` contains the paper's seven standalone baseline adapters.
- NL-SHADE-LBC and SLPSO-ARS are selected through
  `CED_schedule/main_recent_algorithms.cpp`.
- The nine baselines are matched-budget CED adaptations. They are not claimed
  to reproduce the original papers' benchmark representations or budgets.
- `TNUM=1000000` is the fourth, compact stress-test scale. The canonical
  three-scale baseline remains `1000/10000/100000`.

Create a source-only archive with:

```bash
tools/package_github_release.sh
```

The script excludes builds, raw results, temporary files, old ZIP snapshots,
and oversized third-party source trees. It writes a checksum manifest beside
the archive.
