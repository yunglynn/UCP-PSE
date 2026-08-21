# Paper ablation reproduction guide

## Common contract

All paper ablations use four scales, ten paired seeds 20260616--20260625,
`MAXGEN=50`, `POPSIZE=8`, eight MPI processes for parallel variants, `-O2`,
the same data, exact final verification, and program-reported search time.
Change exactly one studied component at a time. Store the full compile command,
environment, commit ID, stdout, and checksum with every run.

The manuscript contains five ablation sections. Dated 2026-08-08 notes are an
earlier deterministic audit and must not be substituted for the ten-run tables.

## A1: Sparse generative-coding operator

Control: UCP-PSE/SGCO. Treatments: full-vector DE/rand/1/bin, PSO, and GA.
Keep the surrogate policy and parallel framework fixed. Use the comparison
driver's `standard_de`, `standard_pso`, and `standard_ga` choices or a dedicated
operator build with otherwise identical flags. The interpretation is limited
to operator sparsity because evaluation and orchestration remain unchanged.

## A2: Hybrid surrogate

Control: `SURROGATE_ENABLED=1`. Treatment: compile with
`-DSURROGATE_ENABLED=0`. Do not change the operator, UCP, archive, migration,
seed, or timer. The exact-objective treatment is much more expensive at the
million scale; it is nevertheless a valid paper ablation when memory permits.
Final quality must still be exact in both arms.

## A3: Uniform control policy

Control: `-DPAPER_COMPONENT_POLICY=1`. Random-PSE treatment:
`-DTRI_POLICY_RANDOM_ABLATION=1
-DTRI_POLICY_FULL_COMPONENT_RANDOM_ABLATION=1`.
The treatment randomly configures the same candidate actions; it does not add
or remove operators. Record action counts to prove both arms used the intended
action space.

## A4: Archive-assisted prewarming

Control: `-DLOCAL_ARCHIVE_ENABLED=1`. Treatment:
`-DLOCAL_ARCHIVE_ENABLED=0`. Keep MPI and ring exchange enabled. At one million
tasks the compact one-individual representation limits what “without archive”
can mean; report the actual allocation path and never imply a dense archive was
materialized if it was not.

## A5: Parallelization

Control: MPI build with `-DUSE_MPI`, launched by `mpirun -np 8`. Treatment:
serial build without `-DUSE_MPI` and with an eight-individual population. Both
arms use the same total generation count and UCP components. Report speedup as
serial program `Time` divided by parallel program `Time`; do not use shell wall
time. The one-million compact layout may make a dense serial treatment
unsupported; mark it unsupported rather than changing memory semantics.

## Required run record

For every scale/variant/seed, retain:

```text
git_commit, dirty_tree_patch_checksum, compiler_version, compile_command,
mpi_version, host_cpu, host_memory, TNUM, CNUM, ENUM, DNUM, MAXGEN, POPSIZE,
BASE_SEED/CED_SEED, MPI_processes, all CED_* variables, best_solution,
program_search_time, exact_evaluation_count, exit_status, stdout_checksum
```

Use a paired nonparametric test for paired seed results and state the test in
the table caption. Runtime values that round to the same displayed precision
must still be tested using the unrounded logs.
