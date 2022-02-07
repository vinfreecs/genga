.. _SmallBodies:

Model for small bodies collisions
=================================

Simulating small bodies of the Solar System (or other planetary systems) often requires the integration of
a very large number of particles. There are more than 1 million asteroids known in the asteroid belt,
and taking into account also the (unseen) smaller fragments would lead to immense numbers,
exceeding current computational capabilities. But still the presence of all these small bodies is important
for the dynamics of meteoroids: collisions can lead to both break-up events and changes in meteoroid rotation rate.
Together with the Yarkovsky effect and Poynting-Robertson drag, collisions between small bodies can influence the
migration rate of asteroids and thus generate impact events on the Earth or other planets.
An alternative to simulate such collisions with an N-body integrator is to include a collision model,
which uses an average probability that a given small body would collide with another small body :cite:p:`Farinella1998`.
In the next two sections, we describe our model for collisional break-up events and rotation changes of small bodies.


| The model for small bodies collisions can be enabled with the :literal:`Use Small Collisions` parameter in the :ref:`param.dat<ParamFile>` file.
  It is only available for test particles.

| All events of rotation reset or fragmentation are reported in the Fragments file :ref:`FragmentsFile`.

| Sinze the fragmentaion routine creates additional debris particles, the initial memory arrays need to be increased to 
  be able to store these additional particles. That can be done with the :literal:`nFragments`: option in the :ref:`param.dat<ParamFile>` file

The following parameters are relevant for the small body collision model  and can be set in the :ref:`param.dat<ParamFile>` file:

- :literal:`Use Small Collisions`, enable small bodies collisions model.
- :literal:`Asteroid rho`: density of the body in kg/m^3.
- :literal:`Asteroid V`: Asteroid collisional velocity V, in m/s.
- :literal:`nFragments`: Number of additional memory size for debris particles, in particle numbers, (default 0). 
- :literal:`Asteroid minimal fragment radius`, in m, :math:`R_m` value. (default = 0.01 m).
- :literal:`Asteroid fragment remove radius`, in m, :math:`R_{del}` value. (default = 0.01 m).


Collisional break-up
--------------------
Enabled with :literal:`Use Small Collisions = 1` or :literal:`Use Small Collisions = 3`.

Following :cite:p:`Farinella1998`, we use a collisional life-time of

.. math::
    \tau_{col} = 2.0 \cdot 10^7 \sqrt{\frac{R}{1\text{m}}} \text{years},

which is equivalent to a break-up probability of :math:`1/\tau_{col}` per year.
When a small body breaks up, its mass is replaced by a series of fragments with radii between the radius
of the original object :math:`R_0`, and :math:`R_m` is set by :literal:`Asteroid minimal fragment radius`.
We use the following distribution to generate the radii of the new fragments :math:`R_f`:

.. math::
    R_{f} = \left[ \left(R_0^{(n + 1)} - R_m^{(n + 1)} \right) \cdot y +  R_m^{(n + 1)} \right]^{1/(n + 1)},

with :math:`n` = -1.5 and a uniform generated random number :math:`y \in (0,1)`. The mass of each generated
fragment is subtracted from the remaining mass of the original body. The first fragment to exceed the remaining
mass gets the remaining mass.

For the velocity distribution of the fragments, we use a scaling value

.. math::
    u = y \cdot m_f^{-1/6},

with the mass of the fragment :math:`m_f`, a random number :math:`y \in (1-a, 1+a)` and :math:`a = 0.2`
:cite:p:`nakamura1991`.

A constant velocity budget of 31 m/s (based on :math:`V_{budget} = (0.3/N)^{0.5} \times V_{impactor}`,
and :math:`N = 8000`; see :cite:p:`wiegert2015` for details) is distributed to the fragments in proportion to :math:`u`.
A randomly distributed velocity vector with the given length sets the new orbit of the particle.
Typical values of these created fragment velocities are of the order of :math:`\sim0.1-1` m/s.

Fragments with a radius smaller than :math:`R_{def}` are removed to prevent having too many particles in the simulation.
The obliquity of the new particles is generated randomly (between 0 and :math:`\pi`) and for the rotation rate, we take

.. math::
    \omega_f = random \left( \frac{1}{36R_f}, \frac{1}{R_f}\right),

which reproduces the range of rotation rates typically observed for Near Earth Asteroids in the meter
:cite:p:`beechbrown2000` to kilometer :cite:p:`Farinella1998` size range.


Collisional reset of rotation rate and obliquity
------------------------------------------------
Enabled with :literal:`Use Small Collisions = 1` or :literal:`Use Small Collisions = 2`.

For each object at each time step, we calculate the probability that its rotation was reset during the last time step
by a virtual collision with another particle, as a function of its radius :math:`R` and previous rotation rate.
The rotation will be changed if the impactor has an angular momentum at the collision time similar to the rotational
angular momentum of the target object, such that the radius of the radius of the projectile is given as :cite:p:`Farinella1998`

.. math::
    r_{projectile} = \left( \frac{2 \sqrt{2} R \omega \rho_{target} }{5 \rho_{projectile} V}\right)^{1/3} R

which, assuming :math:`\rho_{target}= \rho_{projecile}`, simplifies to

.. math::
    :label: eq:rpro

    r_{projectile} = \left( \frac{2 \sqrt{2} \omega}{5 V}\right)^{1/3} R^{4/3}

V is the typical collisional velocity in the asteroid belt (5000 m/s). The probability of a rotation rate reset
through collision with a large enough projectile is given by

.. math::
    :label: eq:collProb

    \frac{1}{\tau_{rot}} = P_i R^2 \cdot 3.5 \cdot 10^5 \cdot (r_{projectile}) ^{-5/2}

with _math:`P_i` the intrinsic collisional probability for the asteroid belt, 
:math:`2.85\cdot 10^{-18} \text{km}^2 \text{yr}^{-1}` :cite:p:`Farinella1998`. Using this number, and replacing
:math:`r_{projectile}` in equation :eq:`eq:collProb` with the expression in equation :eq:`eq:rpro` results in a
final probability of

.. math::
    \frac{1}{\tau_{rot}} = 1.0 \cdot 10^{-18} \cdot R^{-4/3} \left[ \frac{2 \sqrt{2} \omega}{5 V}\right]^{-5/6}

per second (all values in SI units).
If the rotation rate is reset, then both the obliquity and the new rotation rate are randomized. 
