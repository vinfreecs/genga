.. _Collisions:

Collisions
==========

A collision between two particles happens when the separation :math:`r_{ij}` between two bodies :math:`i` and :math:`j`
gets smaller than the sum of their physical radii :math:`R_i + R_j`. The current version of GENGA treats collisions as
perfect inelastic mergers by forming one single bigger body. During this collision process, linear momentum is conserved,
but not the energy, since a part of the kinetic energy and the potential self-energy is transferred into an internal energy
:math:`U`. Angular momentum is conserved by transferring the angular momentum of the two bodies into the spin of the new body.


When a collision happens, the coordinates of the two involved bodies are reported in the collision file (see :ref:`CollisionsFile`).


| The following parameters are relevant for the collision handling and can be set in the :literal:`param.dat` file:

- :literal:`Collision Precision` in units of a physical radius fraction. (default :math:`1.0^{-4}`)
- :literal:`Collision Time Shift`, in units of a physical radius factor. (default 1.0)

- :literal:`def_MaxColl`

Collision details
-----------------

The position and velocity of the new body is calculated as

.. math::
 
   \mathbf{x}_{new} = \frac{\mathbf{x}_i m_i + \mathbf{x}_j m_j}{m_i + m_j}

   \mathbf{v}_{new} = \frac{\mathbf{v}_i m_i + \mathbf{v}_j m_j}{m_i + m_j}

The spin :math:`\mathbf{S}` of the new body is calculated as

.. math::

   \mathbf{L}_{ij} = \frac{m_i m_j}{m_i + m_j} \left( \mathbf{r}_{ij} \times \mathbf{v}_{ij} \right)

   \mathbf{S}_{new} = \mathbf{S}_i + \mathbf{S}_j + \mathbf{L}_{ij}


The change in the internal energy :math:`U` is calculated as

.. math::

   U = \frac{1}{2} \frac{m_i m_j}{m_i + m_j} v_{ij}^2 - G \frac{m_i m_j}{r_{ij}}

The radius :math:`R` of the new particle is set by conserving the mass and by mixing the densities of the
two particles. 

.. math:: 

   R_{new} = \left( R_i^3 + R_j^3 \right)^{1/3}

The index of the nex body is calculated according to the rules:

 - The index of the more massive body.
 - If both bodies have an equal mass, then take the smaller index of the bodies :math:`i` and :math:`j`


At the end of the collision process, the body :math:`i` is transferred to be the new body, and body :math:`j`
is marked as a ghost particle, which is then removed from the simulation later.


Collisions can happen only during a close encounter process, and are called during the Bulirsh-Stoer integration.
The implementation of the collision can be found in the :literal:`collide` function in the :literal:`directAcc.h` file.


.. _CollisionPrecision:

Collision precision
-------------------

The collision process is resolved during the Bulirsh-Stoer direct integration with discrete time steps. Therefore, a collision is 
generally not detected at the exact collision time, but rather when the two particles already overlap by a small amount. 
Using :literal:`Collision Precision = 1.0`, GENGA uses the coordinate from the Bulirsh-Stoer step when the collision is first detected. 
This must be considered when using the data from the collision file or also directly within the code for further analysis. 

However it is possible to increase the collision precision with the :literal:`Collision Precision` parameter in the :literal:`param.dat`
file. This parameter sets the tolerance of the collision detection. It is set in units of a radius fraction
:math:`\frac{(R_i + R_j) - r_{ij}}{R_i + R_j}`, where :math:`R` is the physical radius and :math:`r_{ij}` the separation between
the two bodies. The precision should not be set smaller than :math:`1.0^{-10}`.
When the exact time of a collision is important, a value of around :math:`1^{-4}` is recommended.
Note that this parameter causes some more iterations in the Bulirsh-Stoer routine and can slightly increase the run time
of a simulation.

In :numref:`figCollision` is shown an example of the influence of the collision precision. 

.. figure:: plots/Collision.png  
   :name: figCollision

   Collision and merging of two bodies. In orange are shown the regular time steps before and after the collision. In thin green are shown
   the internal time steps of the Bulirsh-Stoer close encounter integration. A collision is reported when the two bodies already overlap (green color).
   By using a high collision precision, the collision located is resolved at the exact contact time (red color). 


.. _CollisionTshift:

Backtrace Collisions
--------------------

The :literal:`Collision Time Shift` argument, allows to backtrace a detected collision to a time prior to the real collision time, when the
two bodies were separated by a factor :math:`f` times their physical radii. The factor :math:`f` is set by 
:literal:`Collision Time Shift`. This option is especially useful when more complex collision models than perfect mergers are used.
Backtraced collisions are only calculated when the involved bodies really will collide. If they will miss a collision and undergo
a close flyby, then this option is not activated. This option can be combined with the :literal:`Collision Precision` option. 

Since multiple collisions can occur at a similar time, the backtrace option has to resolve every collision isolated. This can result
in a longer run time of the code. Especially when many collisions occur. 

| Backtraced collisions are reported in the file :ref:`CollisionsTshiftFile`.


In :numref:`figCollision3` is shown an example of a backtraced collision.  

.. figure:: plots/Collision3.png  
   :name: figCollision3

   The real collision (in red color) is backtraced until the time when the two bodies are separated by a factor :math:`f` times the sum of their 
   physical radii. In this example, we use :math:`f = 3`. The location of the backtraced collision is shown is blue color. 


