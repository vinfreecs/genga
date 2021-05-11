Running GENGA
=============


Start GENGA
-----------

Ignore lock.dat file




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
