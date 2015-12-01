# The GENGA Code: Gravitational Encounters with GPU Acceleration.
** Authors: Simon Grimm and Joachim Stadel **


** Institute for Computational Science **
** University of Zurich **
** Switzerland **


GENGA is a hybrid symplectic N-body integratori, designed to integrate planet and planetesimal dynamics in the late stage of planet formation and stability analysis of planetary systems. GENGA is based on the integration scheme of the Mercury code (Chambers 1999), which handles close encounters with very good energy conservation. It uses mixed variable integration when the motion is a perturbed Kepler orbit and combines this with a direct N-body Bulirsch-Stoer method during close encounters. The GENGA code supports three simulation modes: Integration of up to 2048 massive bodies, integration with up to a million test particles, or parallel integration of a large number of individual planetary systems. GENGA is written in CUDA C and runs on all NVidia GPUs with compute capability of at least 2.0. All operations are performed in parallel, including the close encounter detection and the grouping of independent close encounter
pairs.

[A Documentation of GENGA can be found here](https://bitbucket.org/sigrimm/genga/src/tip/Documentation.md)

[A paper describing GENGA can be found here](http://arxiv.org/abs/1404.2324)

 ** News: **

 * Version 3.29: Genga supports now up to 32768 bodies in the massive body integration mode. A new parameter "Maximum encounter pairs" sets the maximu number of close encounters for each body.  
 * Version 3.21: Close Encounters can be reported to a separate file
 * Version 3.17: A calendar file can be used to generate irregular coordinate outputs.
 * Verions 3.15: The multi simulation mode can now have individual time step sizes and an individual number of integration steps for each sub-simulation.
 * Version 3.14: The coordinate outputs can now be buffered on the GPU. This increases the performance when lots of consecutive outputs are written. Use the 'Coordinate output buffer' argument to set the buffer size. Energy outputs within a buffer size are skipped.
 * Verion 3.12: The gas disk can now be started from the 'param.dat' file
 * Version 3.10: The aeGrid can be started from the 'param.dat' file instead from the 'define.dat' file. The aeGrid contains now also a semi-major axis versus inclination grid.
 * Version 3.10: The Rcut and RcutSun parameters are moved to the param.dat file and are called now outer- and inner truncation radius. 
 * Version 3.10: The FormatP, FormatT and FormatS parameters are moved to the param.dat file.

 * The test particle mode supports up to 2048 massive bodies. Use the NmaxTestParticles parameter in define.h to increase this value
