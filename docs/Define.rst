.. _Define:

The define.h file
=================

This file contains default values for parameters in the :literal:`param.dat` file.
It also contains constants and additional parameters for GENGA.
When a value is changed in this file, then GENGA needs to be recompiled.


Memory options
--------------

- | :literal:`def_MaxColl` (integer): Maximum number of collisions per time step that can be stored.
  | When a collision happens, then the details of the collisions are stored in an internal buffer on the
    GPU. After the time, this buffer is transferred to the CPU and the information written into the 
    collision file. The size of this buffer is specified by :literal:`def_MaxColl`. When more collisions
    occur in the same time step, then the integration is stoppen and an error message written. 
  | Increasing this number also increases the total amount of needed memory. 
