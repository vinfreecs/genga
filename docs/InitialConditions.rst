.. _InitialConditions:

The initial conditions
----------------------

| The following parameters are relevant for the initial conditions and can be set in the :literal:`param.dat` file:

- :literal:`Input file`
- :literal:`Input file Format`
- :literal:`Default rho`
- :literal:`Angle units`

The initial conditions must be provided in a file, and the name of this file must be set with the :literal:`Input file` argument.
The file must be a text file and every particle corresponds to a new line in the file. The central mass (Sun) must not be included in the
initial conditions. Values for the central mass can be specified directly in the :literal:`param.dat` file, the position and the velocities of
the central mass are set to the origin (heliocentric coordinates). 

The initial conditions must be a text file and the format must correspond to the values set in the 'param.dat' file (Input file Format: << ... >> ).
The data of each particle has to be written in a new line in text format, and the format of the data must correspond to the values of :literal:`Input file Format`.

.. _InitialConditionsFormat:

The initial conditions format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The format of the initial conditions file must be specified with the :literal:`Input file Format` option.
The different entries must be given between the :literal:`<<` and the :literal:`>>` characters and a blank space must be included between every entry.
All available options are listed below. Most values are optional, but they must contain a complete set of Cartesian or Keplerian elements, e.g a
full set of [x y z vx vy vz] or [a(or P) e inc O w M(orT)]


Possible arguments are:

- t: start time of the simulation, in years (default = 0.0).
- i: index of the body. The default value is the line number in the input file, starting with 0. Indices should be unique.
- m: mass in Solar masses (default = 0.0).
- | r: physical radius in AU. If r is not given or the radius is equal to zero, then the program uses the density to calculate the radius.
  | Note that there are different ways to set a radius or density. (See :ref:`ICExamples`).
- rho: density in g/cm^3; optional. The default value can be set by the :literal:`Default rho` parameter.
- x: x-position in AU (heliocentric).
- y: y-position in AU (heliocentric).
- z: z-position in AU (heliocentric).
- vx: x-velocity in AU/day * 0.0172020989 (heliocentric) (See :ref:`Units`).
- vy: y-velocity in AU/day * 0.0172020989 (heliocentric) (See :ref:`Units`).
- vz: z-velocity in AU/day * 0.0172020989 (heliocentric) (See :ref:`Units`).
- a: semi-major axis in AU.
- P: period in days.
- e: eccentricity.
- inc: inclination in radians or degrees (See :literal:`Angle units`).
- O: (Omega, :math:`\Omega`) longitude of the ascending node, in radians or degrees (See :literal:`Angle units`).
- w: (omega, :math:`\omega`) argument of periapsis, in radians or degrees (See :literal:`Angle units`).
- M: mean anomaly, in radians or degrees (See :literal:`Angle units`).
- T: Time of first transit, in BJD (See :ref:`Units`).
- Sx: x-spin in Solar masses AU^2 / day * 0.0172020989. (default = 0.0) (See :ref:`Units`).
- Sy: y-spin in Solar masses AU^2 / day * 0.0172020989. (default = 0.0) (See :ref:`Units`).
- Sz: z-spin in Solar masses AU^2 / day * 0.0172020989. (default = 0.0) (See :ref:`Units`).
- amin: minimal value of semi major axis range for aecount; optional (default = 0.0). (See :ref:`aeLimits`).
- amax: maximal value of semi major axis range for aecount; optional (default = 100). (See :ref:`aeLimits`).
- emin: minimal value of eccentricity range for aecount; optional, (default = 0.0). (See :ref:`aeLimits`).
- emax: maximal value of eccentricity range for aecount; optional, (default = 1.0). (See :ref:`aeLimits`).
- k2: potential Love number of degree 2, dimensionless (default = 0.0).
- k2f: fluid Love number of degree 2, dimensionless (default = 0.0).
- tau: time lag in day / 0.0172020989 (See :ref:`Units`) (default = 0.0).
- Ic: moment of inertia, dimensionless (See :ref:`Units`) (default = 0.4).
- :literal:`-`: skip column, optional.


.. _ICExamples:

Examples
^^^^^^^^

Example 1::

    format in 'param.dat': << x y z m vx vy vz r >>
    input file:
        x1 y1 z1 m1 vx1 vy1 vz1 r1
        x2 y2 z2 m2 vx2 vy2 vz2 r2
        .
        .
        .
        xn yn zn mn vxn vyn vzn rn

GENGA reads the radii from the initial condition file.

Example 2::

    format in 'param.dat': << x y z m vx vy vz rho >>
    input file:
        x1 y1 z1 m1 vx1 vy1 vz1 rho1
        x2 y2 z2 m2 vx2 vy2 vz2 rho2
        .
        .
        .
        xn yn zn mn vxn vyn vzn rhon

GENGA reads the densities from the initial condition file and computes the radii.


Example 3::

    format in 'param.dat': << x y z m vx vy vz >>
    Default rho = 2.0
    input file:
        x1 y1 z1 m1 vx1 vy1 vz1
        x2 y2 z2 m2 vx2 vy2 vz2
        .
        .
        .
        xn yn zn mn vxn vyn vzn

GENGA uses the default density from the 'param.dat' file and computes the radii.


Example 4::

    format in 'param.dat': << x y z m vx vy vz r rho >>
    Default rho = 2.0
    input file:
        x1 y1 z1 m1 vx1 vy1 vz1 r1 rho1
        x2 y2 z2 m2 vx2 vy2 vz2 r2 rho2
        .
        .
        .
        xn yn zn mn vxn vyn vzn r2 rho2

GENGA reads the radii from the initial condition file. If a radius is set to zero, then GENGA reads the density
from the initial condition file and computes the radius.




.. _Units:

Units
-----

The units in GENGA are chose such that the solar mass :math:`M_\odot = 1`, the gravitational constant :math:`G = 1`
and the distance from the Sun to Earth :math:`AU = 1`.

With :math:`G = 6.67408 \cdot 10^-11 \text{m}^3 \text{kg}^{-1} \text{s}^{-2} = 0.01720209895^2 \text{AU}^3 \text{M}_\odot^{-1} \text{day}^{-2}`,
and the Gaussian gravitational constant :math:`k = 0.01720209895`.

Therefore, we need

.. math::

  G = 0.01720209895^2 \text{AU}^3 \text{M}_\odot^{-1} \text{day}^{-2} = 1.0 \, \text{day'}^{-2}
  
and 

.. math::

  1 \, \text{day'} = \text{day} / 0.01720209895,

That means that all time units must be rescaled with 0.01720209895.



Time
^^^^

| The time units are given now as :math:`[time] = \text{day} / 0.01720209895 =` day'. 
| To convert time from day to day', it must be multiplicated by 0.01720209895.

Example: 1 min = 0.0006944 day = 0.00001194 day'. 

An exception is the time step, it is given in days, and converted internally to code units.

Masses
^^^^^^
Masses are given in Solar masses (:math:`M_\odot`).

| Example: The Sun has a mass of 1 :math:`M_\odot`.
| The Earth has a mass of :math:`3.003 \cdot 10^{-6} M_\odot`.


Distances, Radii
^^^^^^^^^^^^^^^^
All distances and radii are given in Astronomical Units, AU.

| Example: The distance from the Sun to the Earth is 1 AU.
| The Earth radius is :math:`4.2635 \cdot 10^{-5}` AU.

Velocities
^^^^^^^^^^
| [v] = AU / day' = AU / day * 0.01720209895.
| To convert velocities from AU / day to AU / day', they must be divided by 0.01720209895.

| Example: The Earth's orbital velocity is 30 km / s = 0.01720209895 AU / day.
| In GENGA units, this is 1.0 AU / day'.

.. _UnitsSpin:

Spin
^^^^
| [spin] = :math:`M_{\odot} AU^2 / day' =  M_{\odot} AU^2 / day \cdot 0.01720209895`
| To convert the spin from :math:`M_{\odot} AU^2 / day` to :math:`M_{\odot} AU^2 / day'`, it must be divided by 0.01720209895.
| The spin is computed as

.. math::

  \vec{S} = \vec{\Omega} \cdot I,

with the angular rotation rate :math:`\vec{\Omega} = \frac{2 \pi}{P}` the rotation period :math:`P`
(in units of day') and the moment of inertia :math:`I`:

.. math::

  I = I_c m r^2

The parameter :math:`I_c` defines the inner structure of the body. :math:`I_c = 2/5` is a solid sphere
with uniform density.

Example: The Sun has a rotational period :math:`P` of 27.5 days, a mass of 1.0 :math:`M_\odot` and a radius of 0.00464 AU.
We assume :math:`I_c = 0.07`.

.. math::
  S = \frac{2 \pi}{27.5 \text{days} \cdot 0.01720209895} \cdot 0.07 \cdot 1.0 M_{\odot} \cdot (0.00464 AU)^2
  = 0.00002001703 \, M_{\odot} AU^2 / day'


Energy
^^^^^^

Angular momentum
^^^^^^^^^^^^^^^^


Time lag
^^^^^^^^
| The time lag :math:`\tau` is given in units of day' = day / 0.0172020989. 
| To convert :math:`\tau` from day to day', it must be multiplicated by 0.01720209895.

Example: 1 min = 0.0006944 day = 0.00001194 day'. 

The time lag :math:`\tau` is related (approximately) to the tidal quality factor :math:`Q` via 
:cite:p:`EfroimskyLainey2007`:

.. math::
  \tau = \frac{\arctan{(1/Q)}}{2 | n - \omega | },

with the mean motion of the planet :math:`n` and the rotation rate of the star :math:`\omega`. 


Time of first transit
^^^^^^^^^^^^^^^^^^^^^
The time of first transit is given in Barycentric Julian Date (BJD).

Physical constants
^^^^^^^^^^^^^^^^^^
The values of used physical constants are set in the :ref:`define.h<constants>` file.

