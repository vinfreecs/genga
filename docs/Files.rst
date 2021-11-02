Output Files
============

.. _LockFile:

The Lock File: lock.dat
-----------------------

This file is created at the beginning of a GENGA simulation. The behaviour depends on the :literal:`IgnoreLockFile` in the :literal:`define.h` file. 

- | If it is set to 0, then GENGA can not be started again from time step 0.
  | The lock file prevents that output files are overwritten and lost.
  | GENGA can only be started again when the lock.dat file is deleted.
- | If it is set to 1, then GENGA can always be started again from time step 0, and all output files are overwritten. 

Restarting GENGA from a time step > 0 is not affected by the lock file.


The Master File: master.out
---------------------------
The master file contains information about the used hardware and simulation progress. If an error occurs then the master file contains more details about that. The master file is not deleted at a new GENGA start.


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
Depends on Format


The number of digits in the output filenames can be changed with the :literal:`def_NFileNameDigits` parameter in the :literal:`define.h` file.

.. _OutputsPerInterval:

Number of outputs per interval
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When this numer is largen than 1, then at each coordinate output interval, n consecutive outputs are written. 

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

The number of digits in the output filenames can be changed with the :literal:`def_NFileNameDigits` parameter in the :literal:`define.h` file.


.. _EnergyFile:

The Energy Output File: Energy<name>.dat
----------------------------------------
This file contains information about the number of particles, angular momentum and the energy.
At the :literal:`Energy output interval`, set in the :literal:`param.dat` file, a new line in this file is written. The format is the following:

    | time0  N  V  T  LI  U  ETotal  LTotal  LRelativ  ERelativ
    | time1  N  V  T  LI  U  ETotal  LTotal  LRelativ  ERelativ
    | .
    | .
    | .

 with

- time in years
- N: Number of particles
- V: Total potential energy
- T: Total Kinetic energy
- LI: Angular momentum lost at ejections
- U: Inner energy created from collisions, ejections or gas disc
- ETotal: Total Energy
- LTotal: Total Angular Momentum
- LRelativ: (LTotal_t - LTotal_0)/LTotal_0
- ERelativ: (ETotal_t - ETotal_0)/ETotal_0

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
The file contains the following columns:

    | time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj
    | .
    | .
    | .


.. _CollisionsTshiftFile:

The Tshift Collisions File: CollisionsTshift<name>.dat
------------------------------------------------------
See :ref:`CollisionTshift`.

In this file are listed the details of the backtraced collisions between particle i and j.
The collison time shift option can be set by the :literal:`Collision Time Shift` argument in the :literal:`param.dat` file.
This file is only created when :literal:`Collision Time Shift` is used.
The file contains the following columns:

    | time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj
    | .
    | .
    | .


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
It contains the details of each enconter event: 

    | time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Sz
    | .
    | .
    | . 


.. _EjectionFile:

The ejection file: Ejections<name>.dat
--------------------------------------
See :ref:`Ejections`.

This file contains the details of all ejection events, in the format:

    | time index m r x y z vx vy vz Sx Sy Sz case
    | .
    | .
    | .


with: case = -3 for bodies removed at the outher boundary,
and case = -2 for bodies removed at the inner boundary. 


.. _StarFile:

The stellar evolution file: Star<name>.dat
------------------------------------------
See :ref:`Tides`.

This file is only produced when :literal:`Use Tides` or :literal:`Use Rotational Deformation` are enabled.
The file contains the parameters of the star in the format:

    | time mass radius Spin_x Spin_y Spin_z Ic Love-number fluid-Love-number time-lag
    | .
    | .
    | .


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

This file is only created when the model for small bodies collisions :literal:`UseSmallCollisions` in the :literal:`param.dat` file is enabled. The file contains information about fragmentation and rotation reset events.

    | time index m r x y z vx vy vz Sx Sy Sz event
    | .
    | .
    | .

the 'event' indicates the following:
 -  0: rotation rate reset
 - -1: Collision, the particle is destroyed, and it is replaced with new fragments (listed in the next lines with event=1 or event=2 of this file)
 - 1: A new fragment particle. The original body is the last body in this file with event = -1.
 - 2: A new fragment particle. The original body is the last body in this file with event = -1. This body is too small and it is directly removed from the simulation.

Each new created fragment gets a new, increasing,  index number.
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




