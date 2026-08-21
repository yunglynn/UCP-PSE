# Reproduction audit for standalone CED solvers

## Scope rule

The CED adaptations may share only the instance loader, exact decoder and exact
objective.  They must not call `MultiMet`, any Meme/ADE operator, the hybrid
surrogate, policy learning, existing archives, or the existing migration
driver.  A source algorithm's population, offspring generation, parameter
adaptation, environmental selection and restart logic must be implemented in
its own translation unit.

## Representation and scale

The CED random-key vector has

`D = 2*TNUM + 2*TNUM*MOPT_NUM = 12*TNUM` for `MOPT_NUM=5`.

All nine CED solvers use exactly eight individuals, as explicitly specified for
this comparison.  Population-size formulas and population-size reduction from
the source papers are outside the reproduction scope.  For example, MadDE's
author code uses `NP=2*D^2`, which would be 288,000,000 individuals already at
`TNUM=1000`.  Core variation, adaptation and selection remain in scope, while
all results are labelled eight-individual CED reimplementations rather than
exact reproductions of the papers' original experiments.

FCA-G must likewise construct the formal context from the sparse CED relation
graph; a dense `D x D` context is not representable at the benchmark scales.

## Primary-material status

| Algorithm | Primary implementation or full method | Current status |
|---|---|---|
| MadDE | Author MATLAB source in `third_party/reproduction_sources/MadDE/MadDE` | Available |
| QPHH | Author C++ source in `third_party/reproduction_sources/qphh-workflow/code` | Available; pristine files require syntax cleanup before validation |
| DTGP-AM | Author manuscript `third_party/reproduction_sources/papers/DTGP_AM_2025.pdf` | Implemented and tested at four CED scales |
| FCA-G | Author MATLAB/Python repository `third_party/reproduction_sources/FCA-G` | Implemented and tested at four CED scales |
| SoEA | Author Python archive `third_party/reproduction_sources/Soft-Scheduling/SoEA/SoEA.zip` | Implemented and tested at four CED scales |
| HGA | Author Python repository `third_party/reproduction_sources/HGA-HFS` | Implemented and tested at four CED scales |
| AMTSA | IEEE TEVC DOI 10.1109/TEVC.2026.3659072; expired author archive; formal abstract and method description available | Paper-guided single-objective CED mapping implemented and tested; the missing explicit task-split variable is mapped to five-operation spatial/temporal coordination |
| Bi-Population CDE | IEEE TEVC DOI 10.1109/TEVC.2023.3325004 plus the authors' 2025 open follow-up pseudocode | Paper-guided CED block-decomposition implementation tested at four scales |
| DCMA | Full author text is publicly exposed through the paper record; no code located | Cancelled at the user's request; not implemented or tested |

No blocked row may be filled with guessed operators or parameters.  The AMTSA
and Bi-Population CDE rows are explicitly labelled CED mappings: neither may be
reported as a reproduction of the original paper's experimental problem.

## CED mapping decisions for the two paper-guided solvers

Bi-Population CDE decomposes the random-key vector into the cloud/edge choice,
cloud/edge resource, manufacturing sequence and manufacturing-device blocks.
At every generation the better half forms the local population and the other
half forms the global population.  The implementation uses local
current-to-best/1 and global rand/1 differential mutation, independent greedy
selection, `F=0.5`, `CR=0.7`, and global-population regeneration below the
paper's diversity threshold `K_div=0.1`.

AMTSA was designed for a bi-objective agricultural multi-robot problem and its
expired archive is not recoverable.  Its CED mapping preserves only mechanisms
stated in the published method description: an adaptive shift from route
structural optimization to task-splitting optimization, followed by spatial
and temporal splitting targeted at a bottleneck resource.  The first mode
updates CED resource and sequence structures.  The second treats the five
manufacturing operations of a job as coordinated workload units: spatial
splitting redistributes their device keys and temporal splitting rebalances
their sequence keys.  CED's scalar exact objective replaces the original
multiobjective environmental selection.  These unavoidable representation and
objective changes must accompany every reported AMTSA result.
