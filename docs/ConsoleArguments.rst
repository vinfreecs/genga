.. _ConsoleArguments:

Console Arguments
=================

Instead of using the parameter file :literal:`param.dat`, some arguments can also be passed as console arguments.
The console arguments have the highest priority and are overwriting the arguments of the :literal:`param.dat` file. The options are:

- -dt <f>: Time step in days
- -ei <i>: Energy output interval
- -ci <i>: Coordinates output interval
- -I <i>: Number of integration steps
- -n1 <f>: Value of n1
- -n2 <f>: Value of n2
- -ndev <i>: Number of devices to use
- -dev <i>: Device number
- -in <s>: Input file name
- -out <s>: Output name
- -R <i>: Restart a simulation at time step i
- -TP <i>: Test Particle mode i = 0, treat all bodies the same way, i = 1: small bodies don't affect big bodies.
- -M <s>: name of file, containing a list of the directories for the multi simulation mode.
- -Nmin <i>: Minimal number of bodies in the simulation, not including test particles.
- -NminTP <i>: Minimal number of test particles in the simulation.
- -SIO <i>: Order of symplectic integrator, The options are 2, 4 or 6.
- -aeN <s>: Name of the aeCount grid
- -t <f>: Start time of the simulation in years
- -MT <i>: Number of simulations in TTV calculations
- -sl <i>: Symplectic recursion levels
- -sls <i>: Symplectic recursion sub steps
- -collPrec <f>: Collision Precision
- -collTshift <f>: Collision Time Shift

Here i means an integer, f a floating point value, and s a string.

