.. _CreateParticles:

Particles Creation Model (under construction
============================================

This model allows to create particles during a running sumulation. The model can be enabled with the :literal:`Create Particles` parameter 
in the :ref:`param.dat<ParamFile>` file.

| It is only available for test particles.
| All particle creation events are reported in the Fragments file :ref:`FragmentsFile`.

Sinze the particle creation routine creates additional particles, the initial memory arrays need to be increased,
to be able to store these additional particles. That can be done with the :literal:`nFragments`: option in the :ref:`param.dat<ParamFile>` file.

The following parameters are relevant for the particle creation model and can be set in the :ref:`param.dat<ParamFile>` file:

- :literal:`Create Particles`, enable small bodies collisions model.


Particles Creation Model 1
--------------------------
Enabled with :literal:`Create Particles = 1`.

This model sets the orbital elements of the new particles by using random numbers. The following ranges must be specified:

- Semi-Major axis, a
- Eccentricity, e 
- Inclination, inc,
- Longitude of ascending node, Omega
- Argument of perihelion, w
- Mean Anomaly, M
- Mass, m
- Radius, r

Also the particles creation rate per year, must be set. E.g. a particles creation rate per year of 2, would generate in average 2 particles per year.
