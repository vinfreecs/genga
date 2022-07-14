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
At the :literal:`coordinate output interval`, set in the :ref:`param.dat<ParamFile>` file, a new output is written. 

The output file format can be set to text-format or to binary-format with the :literal:`Use output binary files` in the :ref:`param.dat<ParamFile>` file.

The number of digits in the output file names can be changed with the :literal:`def_NFileNameDigits` parameter in the :ref:`define.h<Define>` file.


.. _OutputFileFormat:

Output File Format
^^^^^^^^^^^^^^^^^^
| The values of the output files depend on the :literal:`Output file Format`, set in the :ref:`param.dat<ParamFile>` file.
| Possible options are:

- t: time, in years.
- i: index of the body.
- m: mass in Solar masses.
- r: physical radius in AU. 
- x: x-position in AU (heliocentric).
- y: y-position in AU (heliocentric).
- z: z-position in AU (heliocentric).
- vx: x-velocity in AU/day * 0.0172020989 (heliocentric) (See :ref:`Units`).
- vy: y-velocity in AU/day * 0.0172020989 (heliocentric) (See :ref:`Units`).
- vz: z-velocity in AU/day * 0.0172020989 (heliocentric) (See :ref:`Units`).
- Sx: x-spin in Solar masses AU^2 / day * 0.0172020989. (See :ref:`Units`).
- Sy: y-spin in Solar masses AU^2 / day * 0.0172020989. (See :ref:`Units`).
- Sz: z-spin in Solar masses AU^2 / day * 0.0172020989. (See :ref:`Units`).
- | amin: minimal value of semi major axis range for aecount (See :ref:`aeLimits`).
  | Optional.
- | amax: maximal value of semi major axis range for aecount (See :ref:`aeLimits`).
  | Optional.
- | emin: minimal value of eccentricity range for aecount (See :ref:`aeLimits`).
  | Optional.
- | emax: maximal value of eccentricity range for aecount (See :ref:`aeLimits`).
  | Optional.
- | k2: potential Love number of degree 2, dimensionless.
  | (needed when given as initial contions)
- | k2f: fluid Love number of degree 2, dimensionless.
  | (needed when given as initial contions)
- | tau: time lag in day / 0.0172020989 (See :ref:`Units`).
  | (needed when given as initial contions)
- | Ic: moment of inertia, dimensionless (See :ref:`Units`).
  | (needed when given as initial contions)
- | aec: aecount is the number of time steps since the last coordinate output time in which the particles semi major axis and eccentricity where in the aecount box limits (see :ref:`aeLimits`).
  | Optional.
- | aecT, aecountT is the integrated value of all previous aecount values.
  | Optional.
- | encc: enccountT is the number of time steps since the simulation start in which the particle was in a close encounter with another (massive) particle.
  | Optional.
- | Rc: Critical radius for close encounters, rcrit, in AU
  | Optional.
- | test: value stored in the test arrays.
  | Optional.

When the :literal:`Use output binary files` option is used, then the files contain the same data as described before with the following types:

- t: double, 64bit
- i: int, 32bit
- m: double, 64bit
- r: double, 64bit
- x: double, 64bit
- y: double, 64bit
- z: double, 64bit
- vx: double, 64bit
- vy: double, 64bit
- vz: double, 64bit
- Sx: double, 64bit
- Sy: double, 64bit
- Sz: double, 64bit
- amin: float, 32bit
- amax: float, 32bit
- emin: float, 32bit
- emax: float, 32bit
- k2: double, 64bit
- k2f: double, 64bit
- tau: double, 64bit
- Ic: double, 64bit
- aec: float, 32bit
- aecT: float, 32bit
- encc: unsigned long long, 64bit
- Rc: double, 64bit
- test: double, 64bit

The structure of the coordinate output files depends on the parameters :literal:`FormatS`, :literal:`FormatT`, :literal:`FormatP` and :literal:`FormatO`.
Here we describe the possible choices:


FormatS = 0, FormatT = 0, FormatP = 1, FormatO = 0: Out<name>_<time step>.dat
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this format, at each coordinate output interval, a new file is created which contains all particles.
In the multi simulation mode, each sub simulation folder contain individual files::

	t i1 m1 r1 x1 y1 z1 vx1 vy1 vz1 Sx1 Sy2 Sz1 ... 
	t i2 m2 r2 x2 y2 z2 vx2 vy2 vz2 Sx2 Sy2 Sz2 ...
	.
	.
	.
	t in mn rn xn yn zn vxn vyn vzn Sxn Syn Szn ...


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

FormatS = 1, FormatT = 1, and FormatP = 0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This option is not possible, it is equivalent to FormatS = 0, FormatT = 1 and FormatP = 0


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



.. _aeiFiles:

Keplerian-Elemets Output Files: aei<name>.dat
---------------------------------------------

The Coordinate output files can be uses to generate Keplerian-Elements output files with the  :ref:`KE<KE>` tool.
These files contain the following::

	time i a e inc Omega w Theta E M m r

with

- time: time, in years
- i: index 
- a: semi major axis, in AU
- e: eccentricity:
- inc: inclination, in radians
- Omega: longitude of the ascending node, in radians
- w: argument of periapsis, in radians
- Theta: true anomaly, in radians
- E: eccentric anomaly, in radians
- M: mean anomaly, in radians
- m: mass, in Solar masses
- r: radius, in AU

.. _EnergyFile:

The Energy Output File: Energy<name>.dat
----------------------------------------
This file contains information about the number of particles, angular momentum and the energy.
At the :literal:`Energy output interval`, set in the :ref:`param.dat<ParamFile>` file, a new line in this file is written. The format is the following::

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
- U: Inner energy created from collisions, ejections or gas disk, :math:`M_\odot AU^2 / day^2`
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
The precision of the collision output can be adjusted with the :literal:`Collision Precision` argument in the :ref:`param.dat<ParamFile>` file
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
The collision time shift option can be set by the :literal:`Collision Time Shift` argument in the :ref:`param.dat<ParamFile>` file.
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

This file is only created when the model for small bodies collisions :literal:`UseSmallCollisions` or the particle creation mode :literal:`Create Particles` in the :ref:`param.dat<ParamFile>` file are enabled. The file contains information about fragmentation and rotation reset events::

	time index m r x y z vx vy vz Sx Sy Sz event
	.
	.
	.

the 'event' indicates the following:
 - 0: | rotation rate reset
      | Used in :literal:`UseSmallCollisions` = 1 or 2  mode.
 - -1: | Collision, the particle is destroyed, and it is replaced with new fragments (listed in the next lines with event=1 or event=2 of this file)
       | Used in :literal:`UseSmallCollisions` = 1 or 3 mode.
 - 1: | A new fragment particle. The original body is the last body in this file with event = -1.
      | Used in :literal:`UseSmallCollisions` = 1 or 3 mode.
 - 2: | A new fragment particle. The original body is the last body in this file with event = -1.
      | This body is too small and it is directly removed from the simulation.
      | Used in :literal:`UseSmallCollisions` = 1 or 3 mode.
 - 10: | A new particle is created.
       | Used in :literal:`Create Particles` = 1 mode.


Each newly created fragment gets a new, increasing, index number.
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


.. _PoincareFile:

The Poincare surface of section file: Poincare<name><timeInterval>.dat
-----------------------------------------------------------------------
See :ref:`Poincare`

The file contains the coordinates of the Poincare surface of section::

       time index x vx
       .
       .
       .

The crossing events are written consecutively to the file. After each coordinate output interval, another file is created to reduce the file sizes.

