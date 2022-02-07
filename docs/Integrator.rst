Integrator
==========

GENGA is using a hybrid symplectic integrator :cite:p:`Chambers99`.

The hybrid symplectic method uses a smooth changeover function to transfer the calculation of close encounters from the symplectic to a direct N-body
integrator like the Bulirsch-Stoer method or something similar. This transition must be applied smoothly enough to prevent from too large energy errors.
Therefore a critical radius must be defined to set a threshold between the close encounter phase andt he normal integration phase, and it must be chosen
large enough to ensure a smooth transition.


By using democratic coordinates, the Hamiltonian of a planetary system can be split into three parts:

.. math::
	H = H_{A} + H_{B} + H_{C},

with 

.. math::
	H_{A} = \sum_{i=1}^{N} \left( \frac{p_{i}^{2}}{2m_{i}}  - \frac{G m_{i} m_{\star}}{r_{i\star}} \right) \nonumber \\
	- \sum_{i = 1}^{N} \sum_{j = i+1}^{N} \frac{G m_{i} m_{j}}{r_{ij}} [ 1 - K(r_{ij})],

.. math:: 
	H_{B} = -\sum_{i = 1}^{N}\sum_{j=i +1}^{N} \frac{G m_{i} m_{j} }{r_{ij}} K(r_{ij})

and

.. math::
	H_{C} = \frac{1}{2m_{\star}}\left( \sum _{i =1} ^{N} \mathbf{p}_{i} \right) ^{2},

where the symbol :math:`\star` refers to the central mass, and :math:`K(r_{ij})` is a smooth changeover function ranging from 0 to 1.
The tree parts :math:`H_A`, :math:`H_B` and :math:`H_C` correspond to the Keplerian part, the interaction part and the Sun part of the Hamiltonian,
respectively. 


The limit where the changeover functions is applied is defined by a critical radius :math:`r_{\text{crit}}` of a particle :math:`i`:



.. math:: 
	r_{\text{crit},i}= \max(n1 \cdot R_{H,i}, n2 \cdot dt \cdot v_i).
	:label: eq_rcrit

It depends on two terms, the first contains the Hill radius :math:`R_H`, the second contains the time step :math:`dt` and the velocity :math:`v`
of the particle :math:`i`. The two parameters :math:`n1` and :math:`n2` are typically set to 3 and 0.4.


The integrator needs to search for close encounter pairs at each time step and to sort them into independent close encounter groups.
These groups are then integrated with the Bulirsch-Stoer direct N-body method. Ideally, the close encounter groups consist of only a single pair of bodies,
but it can happen that bodies have multiple close encounter pairs, which need to be linked together in a bigger close encounter group.
In the worst scenario, all bodies are in a close encounter with some neighbouring bodies, and all of them are linked together into a single giant
close encounter group. This scenario is likely to happen, when the particle number density is increased for high resolution scenarios.

.. _n1n2:

The n1 and n2 values
--------------------
The values :literal:`n1` and :literal:`n2` from equation :eq:`eq_rcrit` can be set in the :ref:`param.dat<ParamFile>` file.


.. _Symplecicorder:

The order of the symplectic integrator
--------------------------------------

The order of the symplectic integrator can be set with the :literal:`Order of integrator` parameter in the :ref:`param.dat<ParamFile>` file.
Options are 2, 4 or 6. 

The 4th and 6th order symplectic integrators use the description of :cite:p:`Yoshida1990`.

The higher order integrators work the best for cases with few close encounters. 


.. _precheck: 

Finding close encounter candidates
----------------------------------
During the force calculation, the distance of all pairs of bodies are calculated. During this step, close encounter candidates
are reported to a list when the mutual distance is smaller then the critical radius:

.. math::

	r_{ij}^2 < \text{pc} \, r_{\text{crit}}^2.

The factor :literal:`pc` is a safty factor. It can be set in the :ref:`define.h<Define>` file (default = 3.0). 

After all close encounter candidates are found. The real minimal distance between the particles is calculated, by interpolating between the
time steps. Close encounters are reported when:

.. math::

	r_{ij, min}^2 < \text{cef} \, r_{\text{crit}}^2.

The factor :literal:`cef` is a safty factor. It can be set in the :ref:`define.h<Define>` file (default = 1.0). 



.. _Close_Encounters:

Close Encounters
----------------

The :literal:`Maximum encounter pairs` parameter in the :ref:`param.dat<ParamFile>` file, sets the amount of memory that is allocated to store close
ncounter pairs of each body. When a body has more close encounters that specified here, then the simulation is stopped and an error massa is
written. Setting a larger value of :literal:`Maximum encounter pairs` increases the memory usage of the code. 

