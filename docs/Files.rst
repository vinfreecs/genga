Output Files
============

.. _LockFile:

The Lock File: lock.dat
-----------------------

This file is created at the beginning of a GENGA simulation. The behaviour depens on the :literal:`IgnoreLockFile` in the :literal:`define.h` file. 

- | If it is set to 0, then GENGA can not be started again from time step 0.
  | The lock file prevents that output files are overwritten and lost.
  | GENGA can only be started again when the lock.dat file is deleted.
- | If it is set to 1, then GENGA can always be started again from time step 0, and all output files are overwritten. 

Restarting GENGA from a time step > 0 is not affected by the lock file.


The Master File: master.out
---------------------------
The master file contains information about the used hardware and simulation progress. If an error occurs then the master file contains more details about that. The master file is not deleted at a new GENGA start.


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


The Execution Time File: time<name>.dat
---------------------------------------
This file contains the execution time spent for the corresponding Coordinate Output interval, in seconds. The last line contains the total execution time in seconds.
The first column indicates the time step. This last entry in the file is used for the automated restart (restart timestep = -1).


.. _CollisionsFile:

The Collisions File: Collisions<name>.dat
-----------------------------------------
(See :ref:`Collisions`)

In this file are listed the details of the collisions between particle i and j just after they collide.
The precision of the collision output can be adjuted with the :literal:`Collision Precision` argument in the :literal:`param.dat` file
(See :ref:`CollisionPrecision`). 

    | time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj
    | .
    | .
    | .



.. _FragmentsFile:

The Fragments File: Fragments<name>.dat
---------------------------------------

(See :ref:`SmallBodies`)
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


