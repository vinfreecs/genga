.. _Ejections:

Ejections
=========

The following parameters are relevant for ejections and can be set in the :literal:`param.dat` file:

- :literal:`Inner truncation radius`
- :literal:`Outer truncation radius`

When the distance of a particle gets smaller than :literal:`Inner truncation radius`, or when it gets larger than :literal:`Outer truncation radius`, then the particle is removed from the simulation. 
The disance is calculated as:

.. math::

   r = \sqrt{x^2 + y^2 + z^2}



The energy of the removed particle is calulated and added to the inner energy term of the simulation. 
All ejection events are reported in the :ref:`EjectionFile`. 
