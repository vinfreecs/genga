.. _ParamFile:

Parameters in the param.dat file
================================

All parameters in the :literal:`param.dat` file can be changed with recompiling GENGA.
Most parameters are optional, if they are not included in the :literal:`param.dat` file,
then their default values are used. The default values are set in the :ref:`define.h<Define>` file.

Main parameters
---------------
- | :literal:`Time step in days`, (default = 6.0 days)
  | Should be minimal ca. 30 steps per orbit for the second order integrator.
  | Can be set to negative values for an integration backward in time.

- | :literal:`Output name` (default = "test").
  | All output files will contain this name,

- | :literal:`Energy output interval`, Interval when energy information is written, in units of timesteps, (default = 100).
  | See :ref:`EnergyFile`

    - when set to 0, then no energy output is written.
    - when set to -1, then only the very last energy output is written
      (or when a simulation is stopped). 

- | :literal:`Coordinates output interval`, Interval when output coordinates are written to a file, in units of timesteps, (default = 100).
  | See :ref:`OutFile`

    - when set to 0, then no coordinate output is written.
    - when set to -1, then only the very last coordinate output is written
      (or when a simulation is stopped). 

- | :literal:`Number of outputs per interval`, Can be used to write outputs at a number of consecutive time steps at each output Interval.
  | In units of time steps.
  | See :ref:`OutputsPerInterval`.
  | (default = 1).

- | :literal:`Coordinate output buffer`, can be used to temporarily store outputs in a GPU buffer before transferring the data back
  | to the CPU. When outputs are written frequently, then this option can speed up the data transfer.
  | In units of time steps.  
  | (default = 1)
  | The energy outputs within a buffer size are skipped in this mode.

- | :literal:`Irregular output calendar`, filename of the irregular output calendar file,
  | Can be use to write coordinate and energy outputs at specified times.
  | See :ref:`IrregularOutput`

    - :literal:`-` : nothing happens
    - <calendar file name>: the calendar file is used and irregular output files are written.

- | :literal:`Integration steps`, total number of time steps to run.
  | (default = 1000 time steps)

- | :literal:`Input file`, (default = 'inital.dat')
  | Filename (or path) of the initial conditions file
  | The file must exist.
  | See :ref:`InitialConditions` 

- | :literal:`Input file Format`: Format of the initial conditions file. 
  | (default = << x y z m vx vy vz >>)
  | See :ref:`InitialConditionsFormat`

- | :literal:`Default rho`, value for the densities when no value (no rho and no r) is given in :literal:`Input file Format`:
  | in g/cm^3 (default = 2.0)
  | See :ref:`InitialConditionsFormat`

- | :literal:`Angle units` unit of angles in the initial conditions file, (default = radians)
  | either 'radians' or 'degrees'.
  | affects inc, O, w and M.

- | :literal:`Restart timestep`, can be used to continue a finished simulation or the restart at a given time step
  | - 0: Start a new simulation, old files are overwritten (default). 
  | - > 0: Restart GENGA at this time step. 
  | - -1: Continue at the last written output time step.
  | See :ref:`Restart`

Stellar parameters
------------------
- | :literal:`Central Mass`
  | Mass of the central star, in Solar Masses (default = 1.0) 

- | :literal:`Star Radius`
  | Physical radius of the central star in AU (default = 0.00465475877 AU = Solar radius)

- | :literal:`Star Love Number`
  | Love number of the central star (default = 1.0)

- | :literal:`Star fluid Love Number`
  | Fluid Love number of the central star (default = 1.0)

- | :literal:`Star tau`
  | Time lag for tidal force of the central star (default = 0.0)

- | :literal:`Star spin_x`
  | X- component of the spin of the central star, in :math:`M_{\odot} AU^2 / day \cdot 0.01720209895` (default = 0.0)
  | (See :ref:`UnitsSpin`) 

- | :literal:`Star spin_y`
  | Y- component of the spin of the central star, in :math:`M_{\odot} AU^2 / day \cdot 0.01720209895` (default = 0.0)
  | (See :ref:`UnitsSpin`) 

- | :literal:`Star spin_z`
  | Z- component of the spin of the central star, in :math:`M_{\odot} AU^2 / day \cdot 0.01720209895` (default = 0.0)
  | (See :ref:`UnitsSpin`) 

- | :literal:`Star Ic`
  | Moment of inertia of the central star, dimensionless (I/(MR^2) (default = 0.4)
  | This is used to convert between spin and rotational period.
  | (See :ref:`UnitsSpin`) 


Integrator options
------------------

- | :literal:`n1`: Parameter to set critical radius for close encounters (default = 3.0).
  | See :ref:`n1n2`

- | :literal:`n2`: Parameter to set critical radius for close encounters (default = 0.4).
  | See :ref:`n1n2`


- | :literal:`Use Test Particles`, flag to enable test particle mode (default = 0)
  | See :ref:`TestParticles`, and :literal:`Particle Minimum Mass`.
  
    - 0: full gravity mode, compute force between all pairs of particles.
    - 1: test particle mode, small bodies do not affect other bodies.
    - 2: semi active mode, small bodies do only affect large bodies but not other small bodies.

- | :literal:`Particle Minimum Mass`, threshold mass between small and large particles (default = 0.0)
  | All particles with a smaller mass than this value are treated as test particles (if :literal:`Use Test Particles > 0`).
  | When :literal:`Use Test Particles = 0`, then this parameter has no affect. 

- Symplectic recursion levels = 1
- Symplectic recursion sub steps = 2

- | :literal:`Minimum number of bodies`, (default = 0)
  | When the number of bodies (not including test particles) gets smaller than this number, then the simulation will be stopped. 

- | :literal:`Minimum number of test particles` (default = 0)
  | when the number of test particles gets smaller than this number, then the simulation will be stopped.


- | :literal:`Inner truncation radius`, in AU (default = 0.2)
  | When the distance of a particle to the central mass is smaller than this number, then the particle is removed from the simulation.
  | See :ref:`Ejections`

- | :literal:`Outer truncation radius` in AU (default = 50.0)
  | When the distance of a particle to the central mass is larger than this number, then the particle is removed from the simulation.
  | See :ref:`Ejections`

- Order of integrator = 2


Memory options
--------------

- | :literal:`Maximum encounter pairs` arrays size to store close encounter pairs for each body (default = 512)
  | See :ref:`Close_Encounters`

- | :literal:`Nframents`: Number of additional memory size for debris particles, in particle numbers, (default = 0). 
  | See :ref:`SmallBodies`


Options for the a-e and a-i grid
--------------------------------
See :ref:`aegrid`.

- :literal:`Use aeGrid`, flag to enable the a-e and a-i grids, (default = 0)
  
  - 0: nothing happens
  - 1; a-e and a-i grids are created

- | :literal:`aeGrid amin`, minimal value of the same-major axis dimension, in AU.
  |  (default = 0.0)

- | :literal:`aeGrid amax`, maximal value of the same-major axis dimension, in AU.
  |  (default = 5.0)

- | :literal:`aeGrid emin`, minimal value of the eccentricity dimension.
  |  (default = 0.0)

- | :literal:`aeGrid emax`, maximal value of the eccentricity dimension.
  |  (default = 1.0)

- | :literal:`aeGrid imin`, minimal value of the inclination dimension, in radians.
  |  (default = 0.0)

- | :literal:`aeGrid imax`, maximal value of the inclination dimension, in radians.
  |  (default = 0.1)

- | :literal:`aeGrid Na`, number of points in the same-major axis dimension.
  |  (default = 10)

- | :literal:`aeGrid Ne`, number of points in the eccentricity dimension.
  |  (default = 10)
- | :literal:`aeGrid Ni`, number of points in the inclination dimension.
  |  (default = 10)

- | :literal:`aeGrid Start Count`, starting time step when a-e and a-i grid start, in units of time steps
  |  (grids will start at the next bigger coordinate output step)
  |  (default = 0)

- | :literal:`aeGrid name`, name of the grid files.
  |  (default = A)


Options for the gas disk
------------------------

- Use gas disk = 0
- Use gas disk potential = 1
- Use gas disk enhancement = 0
- Use gas disk drag = 2
- Use gas disk tidal damping = 2
- Gas dTau_diss = 10000
- Gas alpha = 1
- Gas beta = 1
- Gas Sigma_10 = 2000
- Gas Mgiant = 1.0E-4
- Gas file name = -

Non-Newtonian forces
--------------------

- :literal:`Solar Constant` : Solar constant at 1 AU in W / :math:`\text{m}^2` (default 1367.0)

- | :literal:`Use GR`: Flag to enable General Relativity corrections
  | See :ref:`GR`

   - 0 (default): no GR correction
   - 1: use GR Hamiltonian splitting 
   - 2: use implicit midpoint with GR force
   - 3: use GR force directly (not symplectic) 

- | :literal:`Use Tides`: Flag to enable tidal forces 
  | See :ref:`Tides`

   - 0 (default): no tidal forces
   - 1: use tidal forces

- | :literal:`Use Rotational Deformation`: Flag to enable rotational deformation forces 
  | See :ref:`RotationalDeformation`

   - 0 (default): no rotational deformation force
   - 1: use rotational deformation force


- | :literal:`Use force` : Old parameter to enable GR, tidal or rotational deformation.
  | This parameter is outdated, use :literal:`Use GR`, :literal:`Use Tides` or :literal:`Use Rotational Deformation` instead.

   - 0 (default) no force applied
   - 1: Use GR correction with Hamiltonian splitting
   - 2: Use tidal forces
   - 4: Use rotational deformation
   - 3: GR + tidal force
   - 5: GR + rotational deformation
   - 6: Tidal force + rotation deformation
   - 7: GR + tidal force + rotation deformation

- | :literal:`Use Yarkovsky`: Flag for Yarkovsky effect 
  | See :ref:`Yarkovsky`

   - 0 (default): no Yarkovsky effect
   - 1: use Yarkovsky effect :math:`\mathbf{a_Y}` 
   - 2: use time averaged Yarkovsky effect :math:`\frac{da}{dt}`

- | :literal:`Use Poynting-Robertson`: Flag for Poynting-Robertson drag 
  | See :ref:`PRdrag`

   - 0 (default) : no Poynting-Robertson drag
   - 1: use Poynting-Roberston drag :math:`\mathbf{a_{PR}}`
   - 2: use time averaged Poynting-Robertson drag :math:`\frac{da}{dt}` and :math:`\frac{de}{dt}`

- | :literal:`Radiation Pressure Coefficient Qpr`, used in the Poynting-Robertson drag, in general assumed to be 1.
  | See :ref:`PRdrag`
  | default = 1.0


- | :literal:`Asteroid emissivity eps`
  | Thermal emissivity factor :math:`\epsilon`, used in the Yarkovsky effect.
  | See :ref:`Yarkovsky`
  | default = 0.95


- | :literal:`Asteroid density rho`, in kg/ :math:`\text{m}^3`
  | Used in Yarkovsky effect, Poynting-Robertson drag and small bodies collision model
  | See :ref:`Yarkovsky`, :ref:`PRdrag`, :ref:`SmallBodies`
  | default = 3500.0  kg/ :math:`\text{m}^3`
 
- | :literal:`Asteroid specific heat capacity C`, in :math:`\text{J} \, \text{kg}^{-1} \text{K}^{-1}`
  | Used in Yarkovsky effect
  | See :ref:`Yarkovsky`
  | default = 680 :math:`\text{J} \, \text{kg}^{-1} \text{K}^{-1}`


- | :literal:`Asteroid albedo A`, Bond albedo, used for Yarkovsky effect. 
  | See :ref:`Yarkovsky`
  | default = 0.2


- | :literal:`Asteroid thermal conductivity K`, in :math:`\text{W} \, \text{m}^{-1} \text{K}^{-1}`
  | Used for Yarkovsky effect
  | See :ref:`Yarkovsky`
  | default = 2.65 :math:`\text{W} \, \text{m}^{-1} \text{K}^{-1}`

- | :literal:`Use Small Collisions`: Flag to enable model for small bodies collision model 
  | See :ref:`SmallBodies`
  
   - 0 (default): model is not enabled
   - 1: enable rotation reset model and fragmentation model for test particles.
   - 2: enable only rotation reset model for test particles.
   - 3: enable only fragmentation model for test particles.


- | :literal:`Asteroid collisional velocity V`, in m/s, used for small bodies collision model
  | See :ref:`SmallBodies`
  | default = 5000 m/s

- | :literal:`Asteroid minimal fragment radius`, in m , used for small bodies collision model
  | See :ref:`SmallBodies`
  | default = 0.01

- | :literal:`Asteroid fragment remove radius`, in m , used for small bodies collision model
  | See :ref:`SmallBodies`
  | default = 0.01 m


Options for encounters
----------------------

- | :literal:`Report Encounters`, flag to enable encounter information (default = 0).
  | See :ref:`Report_Encounters`.

  - 0: nothing happens.
  - 1: Encounter events between two bodies, with a separation less than :literal:`Report Encounters Radius` times
       the sum of their radii, are reported in the encounters-file. 

- | :literal:`Report Encounters Radius`, used for :literal:`Report Encounters`, (default = 1.0).
  | In units of physical radii.

- | :literal:`Stop at Encounter`, flag to stop simulations when a close encounter between two bodies happens, (default = 0).
  | See :ref:`StopAtEncounter`. 

  - 0: nothing happens.
  - 1: Simulations are stoppen when the separation between two bodies is less than :literal:`Stop at Encounter Radius` times
       the Hill radius.

- | :literal:`Stop at Encounter Radius`, used for :literal:`Stop at Encounter`, (default = 1.0).
  | In units of Hill radii.


Options for collisions
----------------------

- | :literal:`Collision Precision`, in units of a physical radius fraction. (default :math:`1.0^{-4}`)
  | This parameter sets the tolerance of the detected collision time. See :ref:`CollisionPrecision`.
  | :math:`|`:literal:`Collision Precision`:math:`|` can not be smaller than :math:`1.0^{-10}`.

  - :math:`precision` > 0: particles overlap slightly, :math:`r_{ij} < Ri + Rj`, :math:`r_{ij} > (Ri + Rj) \cdot (1 - precision)`
  - :math:`precision` < 0: particles do not overlap, :math:`r_{ij} > Ri + Rj`, :math:`r_{ij} < (Ri + Rj) \cdot (1 + precision)`

- | :literal:`Collision Time Shift`, in units of a physical radius factor (default 1.0).
  | Allows to backtrace collision at a point before the collision, when the bodies are separated by 
  | an increased physical radius. 
  | See :ref:`CollisionTshift`

- | :literal:`Stop at Collision`, flag to stop simulations at the first collision time (default 0).
  | This option is not supported in the multi simulation mode.
  | See :ref:`StopAtCollision` 

  -  0: nothing happens.
  -  1: stop simulation at the first collision time.

- | :literal:`Stop Minimum Mass`, used in :ref:`StopAtCollision`, (default :math:`0.0`)
  | Simulations are only stoppen when **both** bodies have a mass larger than this value.

- | :literal:`Collision Model`, can be used to implement a different collision model
  | The default (0) is used for a perfect merger collision.
  | See :ref:`Collisions`

Other
-----

- | :literal:`Set Elements file name`: file name for the set-element table.
  | See :ref:`SetElements`.

  -  :literal:`-`: no file, set-elements function is not used (default)
  -  < data table file name>: This file is used to read the set-elements data table.

- | :literal:`FormatS`: Output file format for multi simulation run.

  -  0: all simulations write to different files in their sub simulation directories (default).
  -  1: all simulations write to the same file in the master directory. 

- | :literal:`FormatT`: Output file format for time steps.

  -  0: all time steps are written to different files (default).
  -  1: all time steps are written to the same file.

- | :literal:`FormatP`: Output file format for particles.

  -  0: all particles are written to different files.
  -  1: all particles are written to the same file (default).

- | :literal:`FormatO`: Output file format for file names.

  -  0: file names contain time steps (default).
  -  1: file names contain output steps.


- | :literal:`Serial Grouping`: Flag for exact reproducible results.
  | See :ref:`SerialGrouping`

   -  0: nothing happens.
   -  1: enable sorting step for exact reproducible results.

- | :literal:`Do kernel tuning`: Flag to enable the self tuning routing of GPU kernel parameters.
  | (default = 1)
  | See :ref:`tuning`

  - 0: if :ref:`tuningFile` is available, then read kernel parameters from that file.
  - 0: if :ref:`tuningFile` is  not available, then use default values.
  - 1: run kernel tuning at the beginning of the integration and write parameters to :ref:`tuningFile`.

- | :literal:`Do Kick in single precision`: Flag for precision in the kick force calculation
  | See: :ref:`KickFloat` 

   - 0: use double precision in force terms (default)
   - 1: use single precision in force terms

Options for TTVs
----------------

- TTV file name = -
- RV file name = -
- TTV steps = 1
- Print Transits = 0
- Print RV = 0
- Print MCMC = 0
- MCMC NE = 0
- MCMC Restart = 0

