# Standalone CED algorithm reproductions

This directory contains CED-specific reimplementations of published search
algorithms.  It deliberately does not include or link `Multimethod.cpp`,
`MultimethodMeme.cpp`, `RecentSchedulingAlgorithms.cpp`, the hybrid surrogate,
either policy learner, or the alternating-archive search driver.

The only shared CED components are:

1. the benchmark file format and loader in `ced_problem.*`; and
2. the exact scheduling objective `CED_Schedule` in `Problems.cpp`.

Each algorithm owns its population, random-number generator, variation,
selection, parameter adaptation, stopping loop, and MPI communication.  Any
mapping forced by the CED representation or by the benchmark scale is recorded
in `REPRODUCTION_AUDIT.md` and must not be described as an exact reproduction
of the source paper's original benchmark experiment.

