Parameters in the param.dat file
================================

Main parameters
---------------
- | Time step in days
  | Should be minimal ca. 30 steps per orbit for the second order integrator.

- | Output name
  | All output files will contain this name

- | Energy output interval, (in timesteps)
  | (See :ref:`EnergyFile`)
  |  - when set to 0, then no energy output is written.
  |  - when set to -1, then only the last energy output is written. 


- Coordinates output interval = 100
- Number of outputs per interval = 1
- Coordinate output buffer = 1
- Irregular output calendar = -
- Integration steps = 100000

Stellar parameters
------------------
- | Central Mass:
  | Mass of the central star, in Solar Masses (default = 1) 
- | Star Radius:
  | Physical radius of the central star in AU (default = 0.00465475877 = Solar radius)
- | Star Love Number:
  | Love number of the central star (default = 1.0)
- | Star fluid Love Number:
  | Fluid Love number of the central star (default = 1.0)
- Star spin_x = 0.0
- Star spin_y = 0.0
- Star spin_z = 0.0
- Star tau = 0.0


- n1 = 3.0
- n2 = 0.4
- Input file = initialplanet_2048.dat
- Input file Format: << x y z m vx vy vz >>
- Angle units = radians
- Default rho = 2.0
- Use Test Particles = 0
- Particle Minimum Mass = 0.0
- Restart timestep = 0
- Symplectic recursion levels = 1
- Symplectic recursion sub steps = 2
- Minimum number of bodies = 1
- Minimum number of test particles = 0
- Inner truncation radius = 0.5
- Outer truncation radius = 50.0
- Order of integrator = 2

Memory options
--------------

- Maximum encounter pairs = 512
- | Nframents: Number of additional memory size for debris particles, in particle numbers, (default 0). 
  | (See :ref:`SmallBodies`)


- Use aeGrid = 0
- aeGrid amin = 0
- aeGrid amax = 5
- aeGrid emin = 0
- aeGrid emax = 1
- aeGrid imin = 0
- aeGrid imax = 0.1
- aeGrid Na = 10
- aeGrid Ne = 10
- aeGrid Ni = 10
- aeGrid Start Count = 1
- aeGrid name = A
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

- Solar Constant : Solar constant at 1 AU in W / :math:`\text{m}^2` (default 1367.0)
- Use force = 0
- | Use Yarkovsky: Flag for Yarkovsky effect (See :ref:`Yarkovsky`)

   - 0 (default): no Yarkovsky effect
   - 1: use Yarkovsky effect :math:`\mathbf{a_Y}` 
   - 2: use time averaged Yarkovsky effect :math:`\frac{da}{dt}`

- Use Poynting-Robertson: Flag for Poynting-Robertson drag (See :ref:`PRdrag`)

   - 0 (default) : no Poynting-Robertson drag
   - 1: use Poynting-Roberston drag :math:`\mathbf{a_{PR}}`
   - 2: use time averaged Poynting-Robertson drag :math:`\frac{da}{dt}` and :math:`\frac{de}{dt}`

- Radiation Pressure Coefficient Qpr, used in the Poynting-Robertson drag, in general assumed to be 1. (See :ref:`PRdrag`)
- | Asteroid emissivity eps
  | Thermal emissivity factor :math:`\epsilon` (See :ref:`Yarkovsky`)
- | Asteroid density rho, in km/ :math:`\text{m}^3` (See :ref:`Yarkovsky`, :ref:`PRdrag`)
 
- Asteroid specific heat capacity C, in :math:`\text{J} \, \text{Kg}^{-1} \text{K}^{-1}` (See :ref:`Yarkovsky`)
- | Asteroid albedo A
  | Bond albedo (See :ref:`Yarkovsky`)
- Asteroid thermal conductivity K, in :math:`\text{W} \, \text{m}^{-1} \text{K}^{-1}` (See :ref:`Yarkovsky`)

- | Use Small Collisions: Flag to enable model for small bodies collisions (See :ref:`SmallBodies`)
  
   - 0 (default): model is not enabled
   - 1: enable rotation reset model and fragmentation model for test particles.

- Asteroid collisional velocity V, in m/s (See :ref:`SmallBodies`)

other
-----

- Set Elements file name = -
- FormatS = 0
- FormatT = 0
- FormatP = 1
- FormatO = 0
- TTV file name = -
- RV file name = -
- TTV steps = 1
- Print Transits = 0
- Print RV = 0
- Print MCMC = 0
- MCMC NE = 0
- MCMC Restart = 0
- Report Encounters = 0
- Report Encounters Radius = 3
- Stop at Encounter = 0
- Stop at Encounter Radius = 1
- Stop at Collision = 0
- Stop Minimum Mass = 0.0

Options for collisions
----------------------

- | Collision Precision, in units of a physical radius fraction. (default :math:`1.0^{-4}`)
  | This parameter sets the tolerance of the detected collision time. See :ref:`CollisionPrecision`.
  | Can not be smaller than :math:`1.0^{-10}`.

- | Collision Time Shift, in units of a physical radius factor (default 1.0).
  | Allows to backtrace collision at a point before the collision, when the bodies are separated by 
  | an increased physical radius. 
  | (See :ref:`CollisionTshift`)


B
-

- | Serial Grouping: Flag for exact reproducible results.
  | See :ref:`SerialGrouping`

   -  0: nothing happens.
   -  1: enable sorting step for exact reproducible results.

