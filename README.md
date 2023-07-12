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


 ** News: **

 * Version 3.182: Corrected z-component of J2 force
 * Version 3.178: Corrected internal energy calculation of the gas disk
 * Version 3.159: Encounters between test particles can be reported with 'Report Encounters = 2'
 * Version 3.155: Added gas disk boundaries in param.dat file
 * Version 3.151: Added Output file format option
 * Version 3.148: Added solar wind factor
 * Version 3.146: Output files can be written in binary format.
 * Version 3.137: Corrected angular momentum units in the Energyfile to Msun AU^2/day.
 * Version 3.135: Moved KickFloat from defin.h to param.dat.
 * Version 3.134: Included spin evolution for tidal and rotational deformation forces.
 * Version 3.129: Included moment of inertia.
 * Version 3.124: Added 'Do kernel tuning' parameter.
 * Version 3.117: Changed and improved Collision Precision, Units are now in physical radius fraction, instead of time.
 * Version 3.116: Changed treatment of Gas alpha for values other than 1.
 * Version 3.115: The Asteroid options (Yarkovsky and PR-drag) are moved from the 'define.h' file to the 'param.dat' file.
 * Version 3.114: The SERIAL_GROUPING option is moved from the 'define.h' file to the 'param.dat' file.
 * Version 3.107: The GENGA repository has moved from Mercurial to Git, because bitbucket removed mercurial.
 * Version 3.103: The param.dat file has more parameters for the different gas disk effects.
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
 * Version 3.10: The aeGrid can be started from the 'param.dat' file instead from the 'define.h' file. The aeGrid contains now also a semi-major axis versus inclination grid.
 * Version 3.10: The Rcut and RcutSun parameters are moved to the param.dat file and are called now outer- and inner truncation radius. 
 * Version 3.10: The FormatP, FormatT and FormatS parameters are moved to the param.dat file.

