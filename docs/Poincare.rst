.. _Poincare:

Poincare surface of section
===========================
By setting the :literal:`def_poincareFlag` :ref:`define.h<Define>` file to 1, GENGA prints the Poincare surface of section.
It prints the coordinates of x and v when a particle crosses the positive x coordinate.
Note that this works only using the second order integrator, and not for test particles or in the multi simulation mode.
An example of the surface of section  of 32 planetesimals can be found [here](https://www.youtube.com/watch?v=a_4cjXVDEAw).
Also note that by enabling this option, the integration speed is reduced.

The coordinates of the Poincare surface of section are reported in the :ref:`PoincareFile`.
