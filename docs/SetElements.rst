.. _SetElements:

Set-elements function
=====================

**(Attention: Update since version 3.92, units and data structure changed)**

This option can be used to modify the orbital parameters of a body, according to a precomputed data table. The data table must be
provided as a file and the name of the file must be specified with the :literal:`Set Elements file name` in the :ref:`param.dat<ParamFile>` file.
The data table can either include Keplerian elements or cartesian coordinates, but the two sets can not be mixed. 
If cartesian coordinates are used, then the positions must be given in heliocentric coordinates, and the velocities can either be given in
barycentric or heliocentric coordinates. The Keplerian or cartesian coordinate sets must not be complete, they can also contain only a subset. 


All elements are interpolated with a cubic interpolation scheme from the provided data table.
The data table must provide elements for at least four different times. With less than four times, the cubic interpolation can not be done.
The data table must also provide a time entry at or after the end of the simulation.

The length of the file can not be larger than the value of :literal:`def_NSetElementsMax` in :ref:`define.h<constants>` file (10000000).

Data table format
^^^^^^^^^^^^^^^^^
The structure of the data file must be the following::

	number of bodies to modify, 't', element symbol 1, elements symbol 2, ...
	time 1, body 1 element 1, body 1 element 2, ...,
	time 1, body 2 element 1, body 2 element 2, ...,
	.
	.
	.
	time 2, body 1 element 1, body 1 element 2, ...,
	time 2, body 2 element 1, body 2 element 2, ...,
	.
	.
	.

with:

- | The number of bodies 'n', indicates how many bodies will be modified. They are the first 'n' bodies in the initial condition file.
  | The index of the planets can not be set. The order of the planets must correspond to the order of the initial conditions file.
- | time is the time of the elements in years. The time must be included.
- elements can be:

    - a, semi-major axis in AU
    - e, eccentricity
    - i, inclination in radians
    - O, (Omega) longitude of the ascending node in radians
    - w, (omega) argument of periapsis in radians
    - T, epoch time in days
    - m, mass in Solar masses
    - r, radius in AU
    - x, X-position in AU (heliocentric)
    - y, Y-position in AU (heliocentric)
    - z, Z-position in AU (heliocentric)
    - vx, X-velocity in AU/day * 0.0172020989 (heliocentric)
    - vy, Y-velocity in AU/day * 0.0172020989 (heliocentric)
    - vz, Z-velocity in AU/day * 0.0172020989 (heliocentric)
    - vxb, X-velocity in AU/day * 0.0172020989 (barycentric)
    - vyb, Y-velocity in AU/day * 0.0172020989 (barycentric)
    - vzb, Z-velocity in AU/day * 0.0172020989 (barycentric)
    - :literal:`-`, skip that column


An example data file to modify the mass and radius of a single body looks like this::

	1 t m r
	0.0000000000000000 3.0024584e-7  2.7582675517426333e-5
	2.2000000476840000 3.00262253e-7 2.75828934594789e-5
	5.3680003681180004 3.00285888e-7 2.758346952419557e-5
	9.9299211920929995 3.00319926e-7 2.7584299066671142e-5
	.
	.
	.


An example data file to modify the semi-major axis, eccentricity and inclination of the first four bodies looks like the following, where
the columns are: time, semi major axis, eccentricity and inclination::

	4 t a e i
	0    5.49973  3.17077e-05  1.03555e-06
	0    5.70011  3.10758e-05  0.00546965
	0    9.9999   3.09719e-06  0.000956204
	0    11.25    2.33299e-06  0.00194193
	100  5.49963  3.09278e-05  1.00262e-06
	100  5.70002  3.01496e-05  0.00527889
	100  9.99984  6.81447e-06  0.000935937
	100  11.2499  7.15947e-06  0.00190979
	200  5.49954  9.73926e-05  4.8273e-06
	200  5.69991  9.89094e-05  0.00507701
	200  9.99975  7.85054e-06  0.000913772
	200  11.2498  7.8151e-06   0.00187511
	300  5.49943  9.31021e-05  4.63857e-06
	300  5.69978  9.42067e-05  0.00486796
	300  9.99967  1.0987e-05   0.000890745
	300  11.2497  1.12345e-05  0.00183792
	.
	.
	.

