# The GENGA Code: Gravitational Encounters with GPU Acceleration.
** Authors: Simon Grimm and Joachim Stadel **


** Institute for Computational Science **
** University of Zurich **
** Switzerland **

[TOC]

# License #
GENGA is free to use, but when results from GENGA are published, then the following paper has to be referenced: 
The GENGA Code: Gravitational Encounters in N-body simulations with GPU Acceleration (Grimm & Stadel 2014).

# Requirements #

GENGA runs on Nvidia GPUs with compute capability of 2.0 or higher. To be able to use the Code one has to install the Cuda Toolkit. This can be downloaded from https://developer.nvidia.com/cuda-downloads.
We strongly recommend to use the driver version 5.0 or higher to get the full performance and correct results. Especially in version 3 and 4 have been observed rounding errors which can affect the output of a simulation serious. 


# Compilation #
GENGA can be compiled with the Makefile by typing 'make SM=xx' to the terminal, where xx corresponds to the compute capability. For example use 'make SM=20' for compute capability of 2.0, or 'make SM=35' for 3.5. 

# Starting GENGA #
GENGA can be startet with

```
#!cuda

./genga
```
followed by optional arguments listed [here](#markdown-header-console-arguments).
When GENGA is started, it creates a Lock file named lock.dat. This file must be deleted when GENGA is started again without a restart time greater than 0. This prevents from data loss by an accidental relaunch. To ignore the lock file the IgnoreLockFile Flag in the define.h file can be set to 1.


# The param.dat File #
Simulation parameters are specified in the 'param.dat' file. The used parameters are listed here, the order can also be changed. If a line in the 'param.dat' file is missing, then the corresponding default value is taken from the 'define.dat' file.

 * The time step, in days
 * The output name
 * The Energy output interval, in time steps. When it is set to zero, then no Energy outputs are written.
 * The Coordinates output interval, in time steps. When it is set to zero, then no coordinare outputs are written.
 * The number of outputs per Coordinate output interval.
 * The Coordinates output buffer, in time steps. If this is larger than 1, then the coordinate outputs are written only block-wise to increase the performance especially in the multi simulation mode with lots of consecutive outputs. The energy outputs within a buffer size are skipped in this mode.  
 * The file name of irregular coordinarte output calendar. "-" means no irregular outputs. See [here](#markdown-header-irregular-coodinate-output) for more details. 
 * The total number of integration steps
 * The central Mass, in solar masses
 * Value of close encounter parameter n1
 * Value of close encounter parameter n2
 * The Star Radius in AU
 * The Star Love Number
 * The Star fluid Love Number
 * The Star spin_x in Solar masses AU^2 * day / 0.0172020989
 * The Star spin_y in Solar masses AU^2 * day / 0.0172020989
 * The Star spin_z in Solar masses AU^2 * day / 0.0172020989
 * The Star tau (time lag)
 * The Input file name
 * The input file format:

    The possible arguments are listed below, some are optional, the order can be changed. The input file format arguments in the param.dat file must be placed between '<< ' and ' >>' and separated with a space.  


    * x = x-coordinate in AU (heliocentric)
    * y = y-coordinate in AU (heliocentric)
    * z = z-coordinate in AU (heliocentric)
    * m = mass in Solar masses
    * vx = x-velocity in AU/day * 0.0172020989 (heliocentric)
    * vy = y-velocity in AU/day * 0.0172020989 (heliocentric)
    * vz = z-velocity in AU/day * 0.0172020989 (heliocentric)
    * rho = density in g/cm^3; optional, the default value can be specified below.
    * r = physical radius in AU; optional, if r is not given or the radius is equal to zero, then the program uses the density to calculate the radius. Note that you have many ways to input the radius or density. Look at the input file section.
    * Sx = x-spin in Solar masses AU^2 * day / 0.0172020989; optional, the default value is 0.0
    * Sy = y-spin in Solar masses AU^2 * day / 0.0172020989; optional, the default value is 0.0
    * Sz = z-spin in Solar masses AU^2 * day / 0.0172020989; optional, the default value is 0.0
    * i = index of the body; optional, the default value is the line number in the input file. Don't give two bodies the same index.
    * t = start time of the simulation in years, optional. The default is zero.
    * amin = minimal value of semi major axis range for aecount; optional, the default value is 0.0. See [here](#markdown-header-aelimits) for more details
    * amax = maximal value of semi major axis range for aecount; optional, the default value is 100. See [here](#markdown-header-aelimits) for more details
    * emin = minimal value of eccentricity range for aecount; optional, the default value is 0.0. See [here](#markdown-header-aelimits) for more details
    * emax = maximal value of eccentricity range for aecount; optional, the default value is 1.0. See [here](#markdown-header-aelimits) for more details
    * k2 = potential Love number of degree 2
    * k2f = fluid Love number of degree 2
    * tau = time lag
    * \- = skip this column, optional

 * The default value of the density in g/cm^3. If the density is not included in the initial condition file, it can be set here globally for all bodies.
 * Use the Test particle mode: 0: All Bodies are treated as massive bodies; 1: Small bodies does not affect big bodies.
 * The Restart time step. If it's set to a value bigger than 0, then the simulation will continue at the specified time step.
 * The minimal number of bodies in the simulation, not counting test particles. If the number of bodies gets smaller than Nmin, then the simulation will stop.
 * The inner truncation radius in AU, bodies with a separation to the Sun smaller than this are taken out of the simulation. (RcutSun in older versions)
 * The outer truncation radius in AU, bodies with a separation to the Sun larger than this are taken out of the simulation. (Rcut in older versions)
 * The Order of the symplectic integrator. The options are 2, 4 or 6
 * The maximum number of close encounter pairs for each body
 * Use aeGrid, 0 or 1. See [here](#markdown-header-aegrid) for more details
 * Miminal major axis for the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * Maximal major axis for the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * Miminal eccentricity for the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * Maximal eccentricity for the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * Miminal inclination for the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * Maximal inclination for the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * The number of cells in a of the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * The number of cells in e of the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * The number of cells in i of the aeCount grid. See [here](#markdown-header-aegrid) for more details
 * The time step when aeCount starts. (In Detail aeCount will start at the next bigger coordinate output step).
 * The name for the aeCount grid
 * use gas disk: By setting this flag, the [gas disc](#markdown-header-gas-disc) is used.
 * The dissipation time for the gas disc in years
 * The power law exponent for the gas disc, alpha 
 * use additional force, which is specified in the file force.h See [here](#markdown-additional-forces) for more details
 * FormatS:  Output file format for multi simulation run. 0: all simulatios write to different files, 1: all simulations write to the same file.
 * FormatT: Output file format for time steps. 0: all time steps are written to different files, 1: all time steps are written to the same file.
 * FormatP: Output file format for particles. 0: all particles are written to different files, 1: all particles are written to the same file.
 * Report Encounters: When this is set to one, then close encounters with a separation less than a factor times the physical radius of the body are reported to a file.
 * Report Encounters Radius: This option sets the factor of the close encounter separation.

# Console Arguments #
Instead of using the parameter file, some arguments can also be passed as console arguments. The console arguments have the highest priority and are overwriting the arguments of the param.dat file. The options are:

 * \-dt f Time step in days
 * \-ei i Energy output interval
 * \-ci i Coordinates output interval
 * \-I i Number of integration steps
 * \-n1 f Value of n1
 * \-n2 f Value of n2
 * \-dev i Device number
 * \-in s Input file name
 * \-out s Output name
 * \-R i Restart a simulation at time step i
 * \-TP i Test Particle mode i = 0, treat all bodies the same way, i = 1: small bodies don't affect big bodies.
 * \-M s name of file, containing a list of the directories for the multi simulation mode.
 * \-Nmin i Minimal number of bodies in the simulation, not including test particles.
 * \-SIO i Order of symplectic integrator, The options are 2, 4 or 6.
 * \-aeN s Name of the aeCount grid
 * \-t f Start time of the simulation in years 

Here i means an integer, f a floating point value, and s a string. 

# Defined Constants and Flags in the define.h file #

Some Constants are defined as c++ preprocessor directives in the define.h file. After changing one of these constants, the code has to be recompiled.

Here it can also be chosen if the gas disc is included or not.

 * All default parameters for the 'param.dat' file. See [here](#markdown-header-the-param.dat-file) for more details

 * pc f: Factor in Pre-checker, Pairs with rij^2 smaller than pc * rcrit^2 are considered as close encounter candidates
 * MaxColl i: Maximum number of Collisions per time step that can be stored
 * cef f: Close encounter factor, pairs with rij^2 smaller than f * rcrit^2 are considered as close encounter pairs.
 * SERIAL_GROUPING i: By setting this flag, simulations can be exactly reproduced, but the performance can be slower.
 * StopAtEncounter i: When > 0, then the multisimulation mode stops simulations when a close encounter occurres (d < StopAtEncounterRadius * RHill). n2 is ignored for the stopping criterion, but still used for numerical close encounters.
 * StopAtEncounterRadius f: factor to stop simulations at close encounters in multisimulation mode
 * poincareFlag i: By setting this flag, the [Poincare surface of section](#markdown-header-the-poincare-surface-of-section) is used.
 * IgnoreLockFile i: By setting this flag, the lock file is ignored and simulation can always be started again.
 * def_GMax i: Defines the maximum size of close encoutner groups as 2^GMax.


Here i means an integer and f a floating point value. 

# The Initial Conditions #
The initial conditions are read from the file specified in 'param.dat' (Input file = ). The initial conditions must be a text file and the format must correspond to the values set in the 'param.dat' file (Input file Format: << ... >> ). The data of each particle has to be written in a new lines in text format. Don't write the central mass (Sun) in the initial condition file, it is set automatically to the heliocentric origin with a mass specified in the 'parm.dat' file. 

 * Example 1
    
    * Format in 'param.dat': << x y z m vx vy vz r >>
    * input file:
        * x1 y1 z1 m1 vx1 vy1 vz1 r1
        * x2 y2 z2 m2 vx2 vy2 vz2 r2
        * .
        * .
        * .
        * xn yn zn mn vxn vyn vzn rn

    GENGA reads the radii from the initial condition file. 

 * Example 2

    * Format in 'param.dat': << x y z m vx vy vz rho >>
    * input file:
        * x1 y1 z1 m1 vx1 vy1 vz1 rho1
        * x2 y2 z2 m2 vx2 vy2 vz2 rho2
        * .
        * .
        * .
        * xn yn zn mn vxn vyn vzn rhon

    GENGA reads the densities from the initial condition file and computes the radii.

 * Example 3

    * Format in 'param.dat': << x y z m vx vy vz >>
    * Default rho = 2.0
    * input file:
        * x1 y1 z1 m1 vx1 vy1 vz1
        * x2 y2 z2 m2 vx2 vy2 vz2
        * .
        * .
        * .
        * xn yn zn mn vxn vyn vzn

    GENGA uses the default density from the 'param.dat' file and computes the radii.

 * Example 4

    * Format in 'param.dat': << x y z m vx vy vz r rho >>
    * Default rho = 2.0
    * input file:
        * x1 y1 z1 m1 vx1 vy1 vz1 r1 rho1
        * x2 y2 z2 m2 vx2 vy2 vz2 r2 rho2
        * .
        *  .
        * .
        * xn yn zn mn vxn vyn vzn r2 rho2

    GENGA reads the radii from the initial condition file. If a radius set to zero, then GENGA reads density from the initial condition file and computes the radius. 

# The Output Files #
## The Coordinate output files
This file contains the heliocentric positions and velocities, the spin and some information about the orbit and close encounters.
At the coordinate output interval, set in the 'param.dat' file, a new output is written. The structure of the coordinate output files depends on the parameters FormatS, FormatT and FormatP. Here we describe the possible choices: 

### FormatS = 0, FormatT = 0, FormatP = 1: Out<name>_<time step>.dat
In this format, at each coordinate output interval, a new file is created which contains all particles. In the multi simulation mode, each sub simulation folder contain individual files.

    t i1 m1 r1 x1 y1 z1 vx1 vy1 vz1 Sx1 Sy2 Sz1 amin1 amax1 emin1 emax1 aecount1 aecountT1 enccountT1 test1
    t i2 m2 r2 x2 y2 z2 vx2 vy2 vz2 Sx2 Sy2 Sz2 amin2 amax2 emin2 emax2 aecount2 aecountT2 enccountT2 test
    .
    .
    .
    t in mn rn xn yn zn vxn vyn vzn Sxn Syn Szn aminn amaxn eminn emaxn aecountn aecountTn enccountTn testn

 * t is the time in years.
 * i is the particle index. If two particles collide, then the new index is the one from the more massive particle. If both particles have the same mass, then the smaller index is taken.
 * amin, amax, emin and emax are the specified boundaries for the aeCount box, see [here](#markdown-header-aelimits) for more details. At a collision, the same rules as for the index are applied to these values.
 * aecount is the number of time steps since the last coordinate output time in which the particles semi major axis and eccentricity where in the aecount box limits. See [here](#markdown-header-aelimits) for more details.
 * aecountT is the integrated value of all previous aecount values.
 * enccountT is the number of time steps since the simulation start in which the particle was in a close encounter with an other (massive) particle.
 * test can be used for some individual values

### FormatS = 1, FormatT = 0 FormatP = 1: Out<name>.dat
Here the difference is that in the multi simulation mode, the coordinates are not written in the sub simulation folders, but in the main folder. The output files contain all particles from all sub simulations.
### FormatS = 0, FormatT = 1 FormatP = 1: Out<name>.dat
Here all the time steps are written to the same file, containing all time steps and all particles.
### FormatS = 1, FormatT = 1 FormatP = 1: Out<name>.dat
Here all time steps are written to the same file, containing all time steps and all particles from all sub simulations.
### FormatP = 0: Outp< index>.dat
Here all particles are written to different files, containing all time steps. 

## The Energy output file: Energy<name>.dat
This file contains information about the number of particles and the energy. At the energy output interval set in the 'param.dat' file, a new line in this file is written. The format is the following:

    time N V T LI U ETotal LTotal LRelativ ERelativ
    .
    .
    .

 with

 * time in years
 * N: Number of particles
 * V: Total potential energy
 * T: Total Kinetic energy
 * LI: Angulat momentum lost at ejections
 * U: Inner energy created from collisions, ejections or gas disc
 * ETotal: Total Energy
 * LTotla: Total Angular Momentum
 * LRelativ: (LTotal_t - LTotal_0)/LTotal_0
 * ERelativ: (ETotal_t - ETotal_0)/ETotal_0

## The Execution time file: time<name>.dat
This file contains the execution time spent for the corresponding Coordinate Output interval, in seconds. The last line contains the total execution time in seconds.

##The information file: info<name>.dat
This file contains general information about the used parameters and hardware.
This file contains also information about the number of close encounter pairs and close encounter group sizes: It shows the Number of prechecked close encounter pairs, the number of close encounter pairs, and the number of groups of sizes 2,4,8,16,32,64,128,256,512,1024 and 2048.
If an error occurs during the simulation, then in this File is written the last coordinates-Output and an information about the error. 

##The collision file: Collisions<name>.dat
In this file are listed the details of the collisions between particle i and j just before they collide:

    time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj
    .
    .
    .

## The ejection file Ejections<name>.dat
In this file are listed the details of the ejected particles. An ejection happens if the distance of a particle to the central mass is greater than the outer truncation radius (Rcut) value specified in the 'defines.h' file, or smaller than the inner truncation radius (RcutSun) value.

    time index m r x y z vx vy vz Sx Sy Sz case
    .
    .
    .

    with: case = -3 for ejected bodies, and case = -2 for bodies falling into the central mass. 


## The encounter file Encounters<name>.dat
This file is only generated when the 'Report Encounters' argument in the 'param.dat' file is greater than zero. It contains the coordinates before the cloasest encounter between two bodies.

    time indexi mi ri xi yi zi vxi vyi vzi Sxi Syi Szi indexj mj rj xj yj zj vxj vyj vzj Sxj Syj Szj
    .
    .
    .


## The aeGrid file: aeCount<aeGrid name>_<time step>.dat
If the use_aeGrid value in the 'param.h' file is set to one, then at the coordinate output interval, a new aeGrid file is created. This file contains two matrices of the size Na x Ne followed by two matrixes of the size Na x Ni, where Na, Ne and Ni are set in the 'param.dat' file. The first matrix contains the number of time steps which a particle was at the corresponding semi major axis and eccentricity bin, since the last output time
The second matrix contains the same information since the beginning of the simulation.
The next to matrices contain the same for the inclination.

In the multi simulation mode all sub simulations are contributing to the same files.
See [here](#markdown-header-aegrid) for more details.

## The Poincare surface of section file: Poincare<name><timeIntervall>.dat
The file contains the following informations (see [here](#markdown-header-the-poincare-surface-of-section) for more details

    time index x vx
    .
    .
    .

The crossing events are written consecutively the the file. After each coordinate output intervall another file is created to reduce the file sizes.

##The master file: master.out
The master file contains information about the used hardware and simulation progress. If an error occurs than the master file contains more details about that. 

# Irregular coodinate output #
In the 'param.dat' file can be specified the name of a calendar file, using the argument 'Irregular output calendar = <fileName>'
This file must contain line by line the desired output times of the coordinates in years.
When this option is used, then output files with names OutIrr... similar to the orignial output files are created. They contain the coordinate outputs of the 
specified time. When the multi simulation mode is uses, then the time and time-step information is only read from the first sub-simulation and applied to all simulations synchronously.
When the argument of 'Irregular output calendar' is equal to "-", or if this line is missing in the 'param.dat', then no irregular output files are generated
 

# Restart a simulation #
A simulation can be restarted from each coordinate output file, by using the -R time step console argument or specify a restart time in the 'param.dat' file. Before you restart a simulation you can also change things in the param.dat file, but the Output name must be the same as in the original run! To be able to restart a simulation, the corresponding coordinate output file, the corresponding line in the energy file and if used, the corresponding aeGrid file, must exist.
Note that the data in the Energy-, Collisions-, Ejections-, time- and info-files are not deleted and the new data is added at the end of these files. But the coordinate output files are OVERWRITTEN with the new data. By restarting a simulation the values of E0 and the inner Energy are read out from the original run and used again.
One can also use a coordinate output file to start a new simulation run with totally different parameters by using the Output file as a new initial condition file. the Input file Format should then be of the form << t i m r x y z vx vy vz Sx Sy Sz amin amax emin emax - - - - >>

When GENGA is restarted using the console argument "-R", it changes the entry of the Lock file named lock.dat. This file must be deleted or modified when GENGA is restarted again from the same time step. This prevents from data loss by an accidental relaunch. To ignore the lock file the IgnoreLockFile Flag in the define.h file can be set t
o 1.

# aeLimits #
The aeLimits values amin, amax, emin and emax are limits for the semi major axis and eccentricity for an individual particle. The values can be set in the initial condition file. If the corresponding particle spends time in the given area in a-e space, then a counter is increased.
This values can be useful in a stability analysis of planetary systems. 

# aeGrid #
The aeGrid is a semi-major axis versus eccentricity grid, and/or a semi-major axis versus inclination grid, which counts the time steps that particles spend in certain ranges in semi major axis and eccentricity or inclination. The dimensions of the grid can be set in the 'param.dat' file with the aeGrid_amin, aeGrid_amax, aeGrid_emin, aeGrid_emax, aeGrid_imin and aeGrid_imax values, where the semi-major axis 'a' is a positive value in astronomical units, the eccentricity 'e' a value between zero and one and the inclination 'i' is an angle bwtween 0 and pi.
The resolution of the grid can be set by the Na, Ne and Ni values.
In the multi simulation mode all sub-simulations are contributing to the same files.

# The Poincare surface of section #
By setting the parameter ** poincareFlag 1 ** in the define.h file, GENGA prints the Poincare surface of section. It prints the coordinates of x and v when a particles crosses the positive x coordinate.
Note that this works only using the second order Intergrator, and not for test particles or in the multi simulation mode. An example of the surface of section  of 32 planetesimals can be found [here](https://www.youtube.com/watch?v=a_4cjXVDEAw).


# Gas disc #
The gas disc is implemented according Morishima et al. (2010) [From planetesimals to terrestrial planets: N-body simulations including the effects of nebular gas and giant planets](http://arxiv.org/abs/1007.0579). It supports gas drag, Type I migration and drag enhancement for small particles.
The parameters for the gas disc can be set in:

 * define.h:
    * useGas: Use the gas disc or not
    * Gasnr_g: Number of cells in r direction for gas grid
    * Gasnz_g: Number of cells in z direction for gas grid
    * Gasnr_p: Number of cells in r direction for particle grid
    * Gasnz_p: Number of cells in z direction for particle grid
    * h_1: scale height at 1AU for c = 1km/s
    * Sigma_10: surface density at 1AU
    * Mgiant: Maximum mass for gas drag
    * M_Enhance: factor for enhamcement
    * Mass_pl: factor for enhamcement
    * fMass_min: factor for enhamcement 
 * param.dat:
    * Dissipation time for the gas in years
    * power law exponent for the gas disc (must be 1 in the current version of GENGA)

# Test particle mode
The test particle mode should be used in simulation with only a small number of massive bodies and lots of massless test particles. The test particles doesn't interact with other test particles. The maximum number of massive bodies in the test particle mode can be set with the NmaxTestParticles parameter in the file define.h. Increasing the number of massive bodies reduces the possible number of test particles due to limited memory.
The test particle mode can be started with the -TP 1 console argument are by setting the 'use Test Particles' value in the 'param.dat' file to one. 

# Multi simulation mode
The multi simulation mode can be used to simulate a large number of small simulations with up to 16 massive particles. For each simulation a new directory is needed, containing the initial condition file and the param.dat file. Note that not all parameters in the 'param.dat' file can be chosen individually, these are only:

 * the time step
 * the number of integration steps
 * the name
 * the central mass
 * the n1 parameter
 * the n2 parameter
 * the initial condition file
 * the default rho
 * the Nmin value
 * the inner truncation radius
 * the outer truncation radius

All the other parameters are read from the simulation number 0.
To start a multi simulation run, an additional file is needed, which contains a list of all sub simulation directory names. For example:

 * path.dat:
    * sim0000
    * sim0001
    * sim0002
    * .
    * .
    * .

The simulation can then be started with './genga -M path.dat'
The sub simulations can all have a different number of bodies. If a sub simulation contains less particles than specified in the Nmin value, then this specific simulation is stopped, and the total number of simulations gets reduced.
In the multi simulation mode, the indexes of the particles should not be greater than 100.

# Additional Forces #
in the file 'force.h' can be included additional forces. In this kernel, all coordinates have already been converted into heliocentric coordinates.
In the 'param.dat' can be specified a binary number 'Use force' for different forces. The kernel is executed for Use force > 0.
Implemented are the following forces:

 * General Relativity correction: binay number 1
 * Tidal forces: binary number 2
 * Rotational Forces: binary number 4

For example 'Use force = 5' will use GR and Rotational forces, 'Use force = 2' will only use Tidal forces.
