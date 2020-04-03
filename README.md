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

 * Version 3.98: The number of digits in the output file names can be specified.
 * Version 3.92: Set Elements files have new format and cubic interpolation.
 * Version 3.90: Includes minimal number of test particles option.
 * Version 3.84: Requires at least CUDA 9, because of warp shuffle operations.
 * Version 3.83: Performs self tuning for kernel parameters.
 * Version 3.78: Moved stop-at-collision parameters to the param.dat file.
 * Version 3.77: Stop at Encounter arguments are included in the param.dat file.
 * Version 3.75: Angle values can be set in degrees or radians
 * Version 3.70: Genga supports now up to 131072 bodies in the massive body integration mode.
 * Version 3.61: MinMass moved to param.dat file.
 * Version 3.60: Ctrl-C signal is recognized to write the current output and stop the simulation.
 * Version 3.60: Restart time step -1 is introduced, to continue at the last output.
 * Version 3.60: Moved gas surface density to param.dat file.
 * Version 3.57: Colision Coordinates can be reported more precisly.
 * Version 3.56: The test particle mode supports semi massive particles
 * Version 3.48: The multi simulation mode can stop simulation at close encounters when 'StopAtEncounter' in the 'define.dat' file is set to 1
 * Version 3.45: The maximum close encounter group size is increased up to 1048576. It can be increased further by changing the 'def_GMax' parameter in the 'define.dat' file. 
 * Version 3.29: Genga supports now up to 32768 bodies in the massive body integration mode. A new parameter "Maximum encounter pairs" sets the maximu number of close encounters for each body.  
 * Version 3.21: Close Encounters can be reported to a separate file
 * Version 3.17: A calendar file can be used to generate irregular coordinate outputs.
 * Verions 3.15: The multi simulation mode can now have individual time step sizes and an individual number of integration steps for each sub-simulation.
 * Version 3.14: The coordinate outputs can now be buffered on the GPU. This increases the performance when lots of consecutive outputs are written. Use the 'Coordinate output buffer' argument to set the buffer size. Energy outputs within a buffer size are skipped.
 * Verion 3.12: The gas disk can now be started from the 'param.dat' file
 * Version 3.10: The aeGrid can be started from the 'param.dat' file instead from the 'define.dat' file. The aeGrid contains now also a semi-major axis versus inclination grid.
 * Version 3.10: The Rcut and RcutSun parameters are moved to the param.dat file and are called now outer- and inner truncation radius. 
 * Version 3.10: The FormatP, FormatT and FormatS parameters are moved to the param.dat file.

