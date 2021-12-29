Output Files
============

.. _LockFile:

The Lock File: lock.dat
-----------------------

This file is created at the beginning of a GENGA simulation. The behaviour depends on the :literal:`IgnoreLockFile` in the :ref:`define.h<Define>` file. 

- | If it is set to 0, then GENGA can not be started again from time step 0.
  | The lock file prevents that output files are overwritten and lost.
  | GENGA can only be started again when the lock.dat file is deleted.
- | If it is set to 1, then GENGA can always be started again from time step 0, and all output files are overwritten. 

Restarting GENGA from a time step > 0 is not affected by the lock file.


The Master File: master.out
---------------------------
The master file contains information about the used hardware and simulation progress. If an error occurs then the master file contains more details about that. The master file is not deleted at a new GENGA start.


The information file: info<name>.dat
------------------------------------
This file contains general information about the used parameters and hardware.

At the beginning, the file lists all used parameters, the version number and driver information.

After the parameters, the file lists the timinigs of the kernel tuning routine (see :ref:`tuning`).

At each energy output interval the file gives information of the number of close encounter.

- Precheck-pairs: number of close encounter candidates found from the prechecker (see :ref:`precheck`).
- CE: total number of detected close encounter pairs (see :ref:`precheck`). 
- groups: number of sepparate close encounter groups; followed by the number of close encounter groups of size 2,4,8,16,32,64,128,256,512,1024,2048 ...


If an error occurs during the simulation, then in this File is written the last coordinates-Output and an information about the error.




.. _tuningFile:

The tuningParameters.dat file
-----------------------------
See :ref:`tuning`.

If the kernel self tuning is enabled, then this file is created, containing the values of the kernel parameters.
If the kernel self tuning is disabled, then kernel parameters can be read from this file. The later option can be useful 
for performance measurement. 


.. _OutFile:

The Coordinate Output Files: Out<name>.dat
------------------------------------------
This file contains the heliocentric positions and velocities, the spin and some information about the orbit and close encounters.
At the :literal:`coordinate output interval`, set in the :literal:`param.dat` file, a new output is written. The structure of the coordinate output files depends on the parameters :literal:`FormatS`, :literal:`FormatT`, :literal:`FormatP` and :literal:`FormatO`.

The number of digits in the output file names can be changed with the :literal:`def_NFileNameDigits` parameter in the :ref:`define.h<Define>` file.

Here we describe the possible choices:


FormatS = 0, FormatT = 0, FormatP = 1, FormatO = 0: Out<name>_<time step>.dat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this format, at each coordinate output interval, a new file is created which contains all particles.
In the multi simulation mode, each sub simulation folder contain individual files::

	t i1 m1 r1 x1 y1 z1 vx1 vy1 vz1 Sx1 Sy2 Sz1 amin1 amax1 emin1 emax1 aecount1 aecountT1 enccountT1 test1
	t i2 m2 r2 x2 y2 z2 vx2 vy2 vz2 Sx2 Sy2 Sz2 amin2 amax2 emin2 emax2 aecount2 aecountT2 enccountT2 test2
	.
	.
	.
	t in mn rn xn yn zn vxn vyn vzn Sxn Syn Szn aminn amaxn eminn emaxn aecountn aecountTn enccountTn testn

with

- t is the time in years.
- i is the particle index. If two particles collide, then the new index is the one from the more massive particle. If both particles have the same mass, then the smaller index is taken.
- m is the mass of the body in Solar masses.
- r is the physical radius of the body in AU.
- x, y, z are the heliocentric positions in AU.
- vx, vy, vz are the heliocentric velocities in AU/day * 0.0172020989.
- Sx, Sy, Sz are the spin components in Solar masses AU^2 / day * 0.0172020989.
- amin, amax, emin and emax are the specified boundaries for the aeCount box (see :ref:`aeLimits`). At a collision, the same rules as for the index are applied to these values.
- aecount is the number of time steps since the last coordinate output time in which the particles semi major axis and eccentricity where in the aecount box limits (see :ref:`aeLimits`).
- aecountT is the integrated value of all previous aecount values.
- enccountT is the number of time steps since the simulation start in which the particle was in a close encounter with another (massive) particle.
- test can be used for some individual values


FormatS = 1, FormatT = 0 FormatP = 1, FormatO = 0: Out<name>.dat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here the difference is that in the multi simulation mode, the coordinates are not written in the sub simulation folders, but in the main folder.
The output files contain all particles from all sub simulations.


FormatS = 0, FormatT = 1 FormatP = 1, FormatO = 0: Out<name>.dat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here all the time steps are written to the same file, containing all time steps and all particles.


FormatS = 1, FormatT = 1 FormatP = 1, FormatO = 0: Out<name>.dat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here all time steps are written to the same file, containing all time steps and all particles from all sub simulations.


FormatP = 0: Outp< index>.dat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here all particles are written to different files, containing all time steps.


FormatS = 0, FormatT = 0, FormatP = 1, FormatO = 1: Out<name>_<output step>.dat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here the difference is that the output files are not named after the time step, but the output step. When the simulation is interrupted at a time step
in between of two output steps, then a backup file 'Outbackup<name>_<time step>.dat' is created. This backup step can be read by the restart
option -R -1. To restart from a normal output file, the real time step, and not the output step must be chosen.


FormatT = 0, and FormatP = 0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This option is not possible, it is equivalent to FormatT = 1 and FormatP = 0


.. _OutputsPerInterval:

Number of outputs per interval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When this number is larger than 1, then at each coordinate output interval, n consecutive outputs are written. 

For example, when the following numbers are set

- :literal:`Coordinates output interval` = 100 
- :literal:`Number of outputs per interval` = 5,

then the following time steps are written as outputs:
0, 96, 97, 98, 99, 100, 196, 197, 198, 199, 200, ...


.. _IrrOutFile:

The Irregular Coordinate Output Files: OutIrr<name><time>.dat
-------------------------------------------------------------
These files are only created when the output calendar file is used. 
The file contains the same structure as the :ref:`OutFile` but at the time specified in the calendar file. The number in the file
name corresponds to the line in the calendar file.
See :ref:`IrregularOutput`.

The number of digits in the output file names can be changed with the :literal:`def_NFileNameDigits` parameter in the :ref:`define.h<Define>` file.


.. _EnergyFile:

The Energy Output File: Energy<name>.dat
----------------------------------------
This file contains information about the number of particles, angular momentum and the energy.
At the :literal:`Energy output interval`, set in the :literal:`param.dat` file, a new line in this file is written. The format is the following::

	time0  N  V  T  LI  U  ETotal  LTotal  LRelativ  ERelativ
	time1  N  V  T  LI  U  ETotal  LTotal  LRelativ  ERelativ
	.
	.
	.

with

- time in years
- N: Number of particles
- V: Total potential energy , in :math:`M_\odot AU^2 / day^2`
- T: Total Kinetic energy, in :math:`M_\odot AU^2 / day^2`
- LI: Angular momentum lost at ejections, in :math:`M_\odot AU^2 / day`
- U: Inner energy created from collisions, ejections or gas disc, :math:`M_\odot AU^2 / day^2`
- ETotal: Total Energy, in :math:`M_\odot AU^2 / day^2`
- LTotal: Total Angular Momentum, in :math:`M_\odot AU^2 / day`
- LRelativ: (LTotal_t - LTotal_0)/LTotal_0, dimensionless
- ERelativ: (ETotal_t - ETotal_0)/ETotal_0, dimensionless

.. _IrrEnergyFile:

The Irregular Energy Output File: EnergyIrr<name>.dat
-----------------------------------------------------
See :ref:`IrregularOutput`.

This file is only created when the output calendar file is used.
The file contains the same structure as the regular :ref:`EnergyFile`, but at output times, specified in the output calendar file.



The Execution Time File: time<name>.dat
---------------------------------------
This file contains the execution time spent for the corresponding Coordinate Output interval, in seconds. The last line contains the total execution time in seconds.
The first column indicates the time step. This last entry in the file is used for the automated restart (restart timestep = -1).


.. _CollisionsFile:

The Collisions File: Collisions<name>.dat
-----------------------------------------
See :ref:`Collisions`.

In this file are listed the details of the collisions between particle i and j.
The precision of the collision output can be adjusted with the :literal:`Collision Precision` argument in the :literal:`param.dat` file
(See :ref:`CollisionPrecision`).
The file contains the following columns::

	time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj
	.
	.
	.


.. _CollisionsTshiftFile:

The Tshift Collisions File: CollisionsTshift<name>.dat
------------------------------------------------------
See :ref:`CollisionTshift`.

In this file are listed the details of the backtraced collisions between particle i and j.
The collision time shift option can be set by the :literal:`Collision Time Shift` argument in the :literal:`param.dat` file.
This file is only created when :literal:`Collision Time Shift` is used.
The file contains the following columns::

	time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj
	.
	.
	.


.. _OutCollisionFile:

The Stop-at-collision-file: OutCollision.dat
--------------------------------------------
See :ref:`StopAtCollision`.

This file is only created when the :literal:`Stop at Collision` option is enabled. It contains all particles of the simulation at the time
when the first collision occurred. The file contains the same columns as the normal output files. 
 

.. _EncounterFile:

The encounter-file: Encounters<name>.dat
----------------------------------------
See :ref:`Report_Encounters`.

This file is only created when the :literal:`Report Encounters` option is enabled.
It contains the details of each encounter event:: 

	time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Sz
	.
	.
	. 


.. _EjectionFile:

The ejection file: Ejections<name>.dat
--------------------------------------
See :ref:`Ejections`.

This file contains the details of all ejection events, in the format::

	time index m r x y z vx vy vz Sx Sy Sz case
	.
	.
	.


with: case = -3 for bodies removed at the outer boundary,
and case = -2 for bodies removed at the inner boundary. 


.. _StarFile:

The stellar evolution file: Star<name>.dat
------------------------------------------
See :ref:`Tides`.

This file is only produced when :literal:`Use Tides` or :literal:`Use Rotational Deformation` are enabled.
The file contains the parameters of the star in the format::

	time mass radius Spin_x Spin_y Spin_z Ic Love-number fluid-Love-number time-lag
	.
	.
	.


.. _IrrStarFile:

The Irregular stellar evolution file: StarIrr<name>.dat
-------------------------------------------------------
See :ref:`Tides`.


This file is only created when the output calendar file is used.
The file contains the same structure as the regular :ref:`StarFile`, but at output times, specified in the output calendar file.


.. _FragmentsFile:

The Fragments File: Fragments<name>.dat
---------------------------------------
See :ref:`SmallBodies`.

This file is only created when the model for small bodies collisions :literal:`UseSmallCollisions` in the :literal:`param.dat` file is enabled. The file contains information about fragmentation and rotation reset events::

	time index m r x y z vx vy vz Sx Sy Sz event
	.
	.
	.

the 'event' indicates the following:
 -  0: rotation rate reset
 - -1: Collision, the particle is destroyed, and it is replaced with new fragments (listed in the next lines with event=1 or event=2 of this file)
 - 1: A new fragment particle. The original body is the last body in this file with event = -1.
 - 2: A new fragment particle. The original body is the last body in this file with event = -1. This body is too small and it is directly removed from the simulation.

Each newly created fragment gets a new, increasing,  index number.
This file permits to recronstruct the collision and fragmentation history of every particle.


.. _aeCountFile:

The a-e and a-i grid files: aeCount<grid name><time step>.dat
-------------------------------------------------------------
See :ref:`aegrid`.

These files are only written when the aegrid option is used.
The files contain four matrices, separated by a blank line. 

- a-e counts since the last coordinate output, size Na x Ne
- a-e counts since the beginning of the simulation, size Na x Ne
- a-i counts since the last coordinate output, size Na x Ni
- a-i counts since the beginning of the simulation, size Na x Ni




