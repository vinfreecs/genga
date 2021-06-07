Running GENGA
=============


Start GENGA
-----------

Before running GENGA, an initial conditions file must be provided, as described in :ref:`InitialConditions`.
An example of an initial conditions file with 2048 planetesimals is provided in the GENGA repository (initialplanet_2048.dat).

All relevant user parameters can be set in the :literal:`param.dat` file, as described in :ref:`ParamFile`.
An example is also provided in the source directory.

GENGA can be started with::

    ./genga [options]

where :literal:`[options]` are optional user parameters as described in :ref:`ConsoleArguments`.

When GENGA is started again in the same directory, then all files are overwritten with the new simulation data.
To prevent GENGA from overwriting data, the lock.file option can be used (See :ref:`IgnoreLockFile`).


.. _Restart:

Restart GENGA to continue simulations
-------------------------------------

A simulation can be restarted from each coordinate output file by using the :literal:`-R` time step console argument or by setting
a restart time in the 'param.dat' file. Before restarting a simulation things in the param.dat file can be changed if necessary,
but the Output name must be the same as in the original run. To be able to restart a simulation, the corresponding coordinate output file,
the corresponding line in the energy file and the time file, and if used, the corresponding aeGrid file, must exist.
Note that the data in the Energy-, Collisions-, Ejections-, time- and info-files are not deleted and the new data is added at the end of these files.
But the coordinate output files are OVERWRITTEN with the new data. By restarting a simulation the values of E0 and the inner Energy are read from
the original run and are reused.
One can also use a coordinate output file to start a new simulation run with totally different parameters by using the Output file as a new initial
condition file. The :literal:`Input file Format` should then be of the form << t i m r x y z vx vy vz Sx Sy Sz amin amax emin emax - - - - >>

When GENGA is restarted using the console argument "-R", it changes the entry of the Lock file named lock.dat. This file must be deleted or modified
when GENGA is restarted again from the same time step. This prevents from data loss by an accidental relaunch. To ignore the lock file, the IgnoreLockFile
flag in the define.h file can be set to 1.

With the restart time -1, GENGA searches automatically for the last output and restarts directly from there. To determine the last output, the last entry
in the time file is used.



.. _TestParticles:

Using test particles
--------------------
The default mode of GENGA computes the gravitational force between all pairs of particles, which leads to :math:`N^2` force calculations.
When :math:`N` is large, then this operation is the dominant part of the entire run time. 

When small particles are used in a simulation, then it can be useful to reduce the amount of mutual force calculations between small particles. 
This can be done with the test particle mode option :literal:`Use Test Particles` in the :literal:`param.dat` file, or by using the
console argument :literal:`-TP 1` or :literal:`-TP 2`.

Test particles are bodies which have a smaller or equal mass than the value specified in the :literal:`Particle Minimum Mass` parameter. 


GENGA supports two different test particles modes, which are described below and visualized in :numref:`figTestParticles`

Test particles mode 1
^^^^^^^^^^^^^^^^^^^^^

:literal:`Use Test Particles = 1`
Small particles (with a mass smaller than :literal:`Particle Minimum Mass`, do not interact with other particles (small and large).
Large particles interact with all other large particles, and affect small particles. 
Small particles can collide with large bodies, but they do not perturb them gravitationally. 


Test particles mode 2 (semi active)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Small particles (with a mass smaller than :literal:`Particle Minimum Mass`, do not interact with other small particles.
Small particles interact with all large particles.
Large particles interact with all large and small particles.



.. figure:: plots/TestParticles.png  
   :name: figTestParticles

   Calculated force terms of the different test particles modes, (0, 1 and 2) for an example of two large and two small particles


.. _IrregularOutput:

Irregular output times
----------------------
When output data is needed on an irregular interval, then the :literal:`Coordinates output interval` and :literal:`Energy output interval`
parameters are not useful. Insted a calendar file with the desired output times can be provided. The name of this file must be set in the
:literal:`Irregular output calendar` parameter in the :literal:`param.dat` file.

The file must contain line by line the times of the desired outputs in units of years. At each irregular output time,
and entry in the irregular energy file is written, and a new irregular output file is created (:ref:`IrrOutFile` and :ref:`IrrEnergyFile`).
 

For example, a calendar file containing the following lines::

	0.1
	0.11
	0.3 

creates three irregular output files OutIrrtest_000000000000.dat, OutIrrtest_000000000001.dat and OutIrrtest_000000000002.dat.

When starting a new simulation, then old OutIrr<name>.dat files and EnergyIrr<name>.dat files are not deleted. If the EnergyIrr<name>.dat file already exists, then the initial energy for a new simulation is read from this file.

When the multi simulation mode is used, then the time and time-step information is only read from the first sub-simulation and applied to all simulations synchronously.

The number of digits in the output filenames can be changed with the :literal:`def_NFileNameDigits` parameter in the :literal:`define.h` file.



.. _tuning:

Use self tuning kernel parameters
---------------------------------

GPU kernels need to be configured with kernel parameters. These are the number of threads per threadblock and the number of threadblocks.
The performance of a GPU code can depend on the specified parameters. Also depending on the used initial conditions and the used GPU type,
the best choice of the kernel parameters can be different. 
Therefore GENGA uses a self tuning routine to determine the best choice at the beginning of the simulations. The used parameters
are reported in :ref:`tuningFile`. If the tuning routine is not enabled, then this file can be used to set the parameters.
If the file does not exist and if the tuning routine is enabled, then default values are used for the parameters.


If :ref:`SerialGrouping` is used, then always the default values are used. The reason is that a different choice of kernel parameters
can lead to a different rounding error array summations. 

When the performance of GENGA measured with a profiling tool (e.g nvprof or nsys) then, GENGA should be run first with the tuning routine
enabled, to write the :ref:`tuningFile`. And in a second step, the profiling can be done without the tuning routine. The reason for this
is that the tuning routine is running some kernels many times, which affects the profiling statistics. 


The tuning routine can be enabled with the :literal:`Do kernel tuning` parameter in the :literal:`param.dat` file.



.. _GengaGL:


GengaGL: Real time visualization with openGL
--------------------------------------------

GENGA offers the option to view a real time visualization of a simulation by using openGL. Thi option can only be used with a GPU that is 
connected directly to the monitor. Typically, GPUs in a computer cluster can not be used for this. GENGA uses the CUDA openGL interoperability 
to visualize the simulated particle directly with the GPU. No data transfer to the CPU is needed, therefore only very little extra run time is
added to the simulation. 


To compile GENGA with openGL, GLUT must be installed. In Ubuntu this can be installed with the freeglut3-dev package. 

The GENGA - openGL interoperability code is included in the GengaGL directory and can be compiled with the provided Makefile in the same way as
the original GENGA code.

GengaGL can be started with::

	./gengaGL [options]

When GengaGL is started, then a window with the visualization is created.
At the origin, the axis of the coordinate system is plotted.
On the bottom left, the time of the integration is shown. On the bottom right, the scale of the coordinate system axis is indicated.
The color of the particles correspond to the masses of the bodies. Red indicates the largest masses, yellow the smallest masses.
Massless particles are plotted in white color. 


The following options can be used to interact with the visualization:

- Left click and move the mouse to rotate around the z-axis and the y-axis.
- Right click and move the mouse up or down to change the rotation speed of the reference frame.
- Press 'a' + left click and move the mouse to shift the origin.
- Press space, 'p' or middle click to pause the simulation. Press again to continue.
- Use the scroll option to zoom in and out of the visualization.
- Press 'r' to increase the point size of the particles.
- press 't' to decrease the point size of the particles.
- Press 'o' to reset to the original visualization settings.



In :numref:`figGengaGL` is shown a screenshot of GengaGL.

.. figure:: plots/gengaGL.png  
   :name: figGengaGL

   Screenshot of the GENGA real time visualization tool using openGL. Shown is the inner part of the Solar System, including
   Jupiter and the asteroid belt. The trojans are nicely visible on the Lagrange points of Jupiter.
