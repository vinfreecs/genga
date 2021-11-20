.. _Define:

Parameters in the define.h File
===============================

This file contains all default values of the parameters in the :literal:`param.dat` file.
It also contains constants and additional parameters for GENGA.
When a value is changed in this file, then GENGA needs to be recompiled.


.. _OldShuffle:

Use old Shuffle
---------------
CUDA has replaced the shuffle functions :literal:`__shfl_xor` to a new function :literal:`__shfl_xor_sync`.
GENGA offers a flag to switch between the two versions.
If an old CUDA version is used (< CUDA 9.0) then the :literal:`def_OldShuffle` parameter in the :literal:`define.h` file must be set to 1.
 
- 0: use new sync shuffle version (default)
- 1: use old shuffle version


.. _IgnoreLockFile:

Ignore Lock File
----------------

- | :literal:`def_IgnoreLockFile`, flag to set the behaviour of starting (not restarting) a new simulation
    - 0: When previous output files exist, then GENGA can only be started again after deleting the :literal:`lock.dat` file.
         This option prevents that existing data is overwritten. 
    - 1: GENGA can be started even if previous files exist. Previous files are deleted and overwritten.


Memory options
--------------

- | :literal:`def_MaxColl` (integer): Maximum number of collisions per time step that can be stored.
  | When a collision happens, then the details of the collisions are stored in an internal buffer on the
    GPU. After the time, this buffer is transferred to the CPU and the information written into the 
    collision file. The size of this buffer is specified by :literal:`def_MaxColl`. When more collisions
    occur in the same time step, then the integration is stoppen and an error message written. 
  | Increasing this number also increases the total amount of needed memory.
  | (default = 120)


- | :literal:`def_SLevelsMax` (integer): Define the maximum number of symplectic substep levels. 
  | Increasing this number will increase the amount of needed memory.


Other
-----
- :literal:`def_tol`: Tolerance in Bulirsch Stoer integrator (default = 1.0e-12). 
- :literal:`def_dtmin`: Minimal time step in Bulirsch Stoer integrator (default = 1.0e-17).
- :literal:`def_NFileNameDigits`: set number of digits in the names of the output files (default = 12).

- | :literal:`def_pc`: Factor in Pre-checker, pairs with :math:`r_{ij}^2 < \text{pc} \, r_\text{crit}^2` are considered as close encounter candidates.
  | (default = 3.0)
  | See :ref:`precheck`.

- | :literal:`def_cef`: Factor in close encounter detector, pairs with :math:`r_{ij}^2 < \text{cef} \, r_\text{crit}^2` are considered as close encounter paris.
  | (default = 1.0)
  | See :ref:`precheck`.

- | :literal:`def_GMax`: Defines the maximum size of close encounter groups as :math:`2^\text{GMax}`.
  | (default = 20)

- | :literal:`def_poincareFlag`: Flag to enable the Poincare surface of section calculation
  | See :ref:`Poincare`

  - 0: Poincare surface of section is not calculated
  - 1: Poincare surface of section is calculated


.. _constants:

Physical constants
------------------
The following physical constants are used by GENGA and can be changed in this section of the the :literal:`define.h` file:


- | :literal:`def_ksq = 1.0`
  | Squared Gaussian gravitational constant in current units

- | :literal:`def_Kg 2.959122082855911e-4`
  | Squared Gaussian gravitational constant in :math:`AU^3 day^{-2} M_\odot^{-1}` , used for conversion

- | :literal:`dayUnit 0.01720209895`

- | :literal:`def_AU 149597870700.0`
  | AU in m

- | :literal:`def_Solarmass 1.98855e30`
  | Solar Mass in Kg

- | :literal:`def_c 299792458.0`
  | speed of light in m/s

- | :literal:`def_cm 10065.3201686`
  | speed of light in AU / day * 0.0172020989     

- | :literal:`def_sigma 5.670373e-8`
  | Stefan Boltzmann constant in :math:`J m^{-2} s{^-1} K{^-4}`

