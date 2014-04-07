# The GENGA Code: Gravitational Encounters with GPU Acceleration.
** Authors: Simon Grimm and Joachim Stadel **


** Institute for Computational Science **
** University of Zurich **
** Switzerland **


GENGA is a hybrid symplectic N-body integratori, designed to integrate planet and planetesimal dynamics in the late stage of planet formation and stability analysis of planetary systems. GENGA is based on the integration scheme of the Mercury code (Chambers 1999), which handles close encounters with very good energy conservation. It uses mixed variable integration when the motion is a perturbed Kepler orbit and combines this with a direct N-body Bulirsch-Stoer method during close encounters. The GENGA code supports three simulation modes: Integration of up to 2048 massive bodies, integration with up to a million test particles, or parallel integration of a large number of individual planetary systems. GENGA is written in CUDA C and runs on all NVidia GPUs with compute capability of at least 2.0. All operations are performed in parallel, including the close encounter detection and the grouping of independent close encounter
pairs.

[A Documentation of GENGA can be found here](src/[tip]/Documentation)
