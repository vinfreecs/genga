.. _Encounters:

Encounters
==========

This section describes the options or reporting encounter events. It is not related to the close encounter arguments used for the changeover function of the integrator. 


| The following parameters are relevant for the encounter-report or stop-at-encounters handling and can be set in the :ref:`param.dat<ParamFile>` file:

- :literal:`Report Encounters`
- :literal:`Report Encounters Radius`
- :literal:`Stop at Encounter`
- :literal:`Stop at Encounter Radius`



.. _Report_Encounters:

Report Encounters
-----------------
An encounter event is defined as the moment when the distance between two bodies has reached a minimum. Typically an encounter event happens only once per orbit.
Further, only encounter events are considered whith


.. math::
 
   r_{ij} < f * R_i + f * R_j,

where :math:`r_{if}` is the separation between the two involved bodies, :math:`R` is the physical radius and the factor :math:`f`
is the value given in :literal:`Report Encounters Radius`. 

Note that the reported coordinates of the encounter events are not exactly refined to the true closest approach, instead the coordinates of the next Bulirsh-Stoer time step is reported.

 
When this closest approach happens within a close encounter integration phase, then it is directly detected by the interpolation polynomial used in :ref:`Close_Encounters`.
When the closest approach happens outside of a close encounter integration, meaning when the value of :literal:`Report Encounters Radius` time the sum of the two physical radii
is larger than the mutual critical radius, then the critical radii are automatically increased 
by this value. As a consequence, the integration can be slowed down if :literal:`Report Encounters Radius` is set too large. We recommend a value between 1 and 100. 

When a collision happens, the coordinates of the two involved bodies are reported in the encounters file (see :ref:`EncounterFile`).


.. _Report_Encounters_Cloud_Size:

Report Encounters Cloud Size
----------------------------

A particle cloud is a collection of particles, representing the same physical object. Encounters between particles belonging to the same cloud are not reported in the encounters file. The cloud index is computed via the particles index and the cloud size as

.. math::

	\text{Cloud index} = (\text{round to integer}) \left( \frac{\text{Particle index}}{\text{Cloud Size}} \right),

where the cloud size is set by the :literal:`Report Encounters Cloud Size` parameter.

Therefore, the particle indices can be used to assign particles to different particle clouds. 
If the cloud size is equal to 1, then all encounters between all particles are reported.


.. _StopAtEncounter: 

Stop at Encounter
-----------------

When this options is enabled, then simulations are stoppen when a close encounter occur with 

.. math::
 
   r_{ij} < g * RH_i + g * RH_j,

where :math:`r_{if}` is the separation between the two involved bodies, :math:`RH` is the Hill radius and the factor :math:`g`
is the value given in :literal:`Stop at Encounter Radius`.





