# The GENGA Code: Gravitational Encounters with GPU Acceleration.
** Authors: Simon L. Grimm and Joachim Stadel **


** Center for Space and Habitability (CSH), **
** University of Bern, **
** Switzerland **


** Institute for Computational Science, **
** University of Zurich, **
** Switzerland **


GENGA is a hybrid symplectic N-body integrator, designed to integrate planet and planetesimal dynamics in the late stage of planet formation and stability analysis of planetary systems. GENGA is based on the integration scheme of the Mercury code (Chambers 1999), which handles close encounters with very good energy conservation. It uses mixed variable integration when the motion is a perturbed Kepler orbit and combines this with a direct N-body Bulirsch-Stoer method during close encounters. The GENGA code supports three simulation modes: Integration of up to 60000 - 100000  massive bodies, integration with up to a million test particles, or parallel integration of a large number of individual planetary systems. GENGA is written in CUDA C and runs on all NVidia GPUs with compute capability of at least 2.0. All operations are performed in parallel, including the close encounter detection and the grouping of independent close encounter
pairs.

[A documentation of GENGA can be found here](https://genga.readthedocs.io/en/latest/)

[A paper describing GENGA can be found here](https://ui.adsabs.harvard.edu/abs/2014ApJ...796...23G)

[The GENGA II paper can be found here](https://ui.adsabs.harvard.edu/abs/2022arXiv220110058G)


 ** GENGA Tutorial **
Try GENGA in Google colab [with this tutorial](https://gist.github.com/sigrimm/93e2faed18e0e39e82aa226097b78e2c) .

Open the tutorial notebook in Colab and learn how to run GENGA.
The same notebook is also included in this repository here: GengaTutorial.ipynb .


## About this fork

This is a fork of GENGA for one kind of simulation: **a few massive bodies (the 4 giant
planets) plus many massless test particles**, integrated for 4.5 Gyr. The original code,
its full version history and its news list are on Bitbucket:
[bitbucket.org/sigrimm/genga](https://bitbucket.org/sigrimm/genga). Please cite the GENGA
papers above when using this code.

### Why the changes

GENGA was designed for many massive bodies, where the O(N^2) force calculation takes almost
all the time. With only 4 massive bodies the force calculation is cheap, and other parts of
the time step become the cost: loops that visit every particle to find the ~20 that are in a
close encounter, a momentum sum over a million bodies of which 4 have mass, and many small
kernel launches per step. The changes below remove that work. They do not change the
integrator or the physics.

### The changes

| change | file | what it does |
|---|---|---|
| group compaction | `Encounter3.h` | `group_kernel` loops over the bodies in the encounter list instead of over every particle. It runs in one block, so the full scan made it the most expensive kernel while encounters are frequent (45-65% of GPU time) |
| BSB source loop | `BSB.h` | the Bulirsch-Stoer force loop in a close encounter group runs over the mass sources only, since test particles exert no force |
| Sun kick sum | `HC.h` | `HC32d1_kernel` sums the momentum over the massive bodies only, and `HC32d2_kernel` is skipped when it has nothing to add. About 12% of the step on GH200 |
| HC32d3 + fg | `FG2.h` | the Sun kick shift and the Kepler drift run in one kernel |
| HC32d1 folds | `Kick3.h`, `HC.h` | the momentum sum is done inside the kick before it and the shift after it, removing two launches per step |
| first half step | `Rcrit.h`, `FG2.h` | kick + Sun kick + drift in one kernel, from a copy of the planets saved at the start of the step |

A step without close encounters went from 11 kernel launches to 5. On GH200, with 34,644 test
particles, the changes measured 1.74x faster at the start of a simulation and 1.27-1.34x in the
relaxed end state, which is most of a 4.5 Gyr run (measured with an earlier version of the
changes, before the last two rows were added).

A fusion of `acc4C_kernel` and `kick32Ab_kernel` was tried and removed: it was 1.9% slower
on GH200, because the fused kernel ran at half the occupancy.

### Switches

All changes sit behind switches in `source/define.h`. Setting all of them to 0 builds the
original code.

| switch | default | |
|---|---|---|
| `def_LongTermSim` | 1 | group compaction, BSB source loop, Sun kick sum |
| `def_FUSE_HC32D3_FG` | 1 | HC32d3 + fg |
| `def_FUSE_HC32D1` | 1 | HC32d1 folds |
| `def_FUSE_MEGA` | 1 | first half step in one kernel |

**Test particle masses must be exactly 0.** GENGA treats a body as a test particle when its
mass is at most `MinMass`, but the changes assume the mass is 0. Check an input with
`awk '{print $1}' <input> | sort -u | head`.

### Correctness

The changes compute the same numbers in fewer steps, so the output should be identical to the
original code byte for byte. This was checked on an RTX 5060 , GH200 and A100 for every change. The one
exception is the BSB source loop, which adds the forces in a different order and can change the
last bit when a group contains 3 or more planets. The energy file does not show errors in test
particles (they have no mass), so the check compares the coordinate output files directly.

### Build

    cd source && make SM=90      # GH200 (JUPITER); SM=80 for A100

`source/Makefile` defaults to `SM=60`, which builds for the wrong GPU.

