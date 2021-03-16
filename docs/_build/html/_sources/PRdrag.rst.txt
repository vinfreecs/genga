.. _PRdrag:

Poynting-Robertson drag
=======================

The Poynting-Robertson drag is implemented according to :cite:p:`Burns1979`
and is available in the schemes (we recommend scheme 1):

- | Velocity kick :math:`\mathbf{a_{PR}}`
  | set :literal:`Use Poynting-Robertson = 1` in the :literal:`param.dat` file.
  | See Equation :eq:`eq_3a` and :eq:`eq_3b`
- | Time averaged change in sami-major axis and eccentricity :math:`\frac{da}{dt}`, :math:`\frac{de}{dt}`
  | set :literal:`Use Poynting-Robertson = 2` in the :literal:`param.dat` file.
  | See Equation :eq:`eq_4`
	

The following parameters are relevant for the Poynting-Robertson drag and can be set in the :literal:`param.dat` file:

- :literal:`Use Poynting-Robertson`
- :literal:`Solar Constant`: Solar Constant at 1 AU in W /m^2
- :literal:`Radiation Pressure Coefficient Qpr` (in general assumed to be 1)


.. math::
   :label: eq_3a

   \frac{da}{dt} = -\frac{\eta}{a}Q_{pr} \frac{(2 + 3e^2)}{(1 - e^2)^{3/2}}

.. math::
   :label: eq_3b

   \frac{de}{dt} = - \frac{5}{2}\frac{\eta}{a^2}Q_{pr} \frac{e}{(1 - e^2)^{1/2}},

.. math::
   :label: eq_4

   \frac{d \mathbf{v}}{dt} = \frac{\eta c}{r^2} Q_{pr} \left[ \left(1 - \frac{\dot{r}}{c} \right) \hat{r} - \frac{\mathbf{v}}{c} \right].


Test of the Poynting-Robertson drag
-----------------------------------

In :numref:`figPRdrag` is shown a test of the Poynting-Robertson drag for the same initial conditions as in 
:ref:`YarkovskyTest`, but with eccentricities :math:`\neq` 0. 


| Relevant parameters for this example:

-  Use Poynting-Robertson = 1 (2)
-  Solar Constant = 1367
-  Radiation Pressure Coefficient Qpr = 1

| Initial conditions:

- Semi-major axis a = 2 AU
- Eccentricity = 0.05
- Inclination = 0
- Argument of perihelion = 0 - 2 :math:`\pi`
- Longitude of ascending node = 0 - 2 :math:`\pi`
- Mean anomaly = 0 - 2 :math:`\pi`
- Density :math:`\rho` = 3500.0 kg/m^3
- Physical radius R =  0.1 - 100 m




.. figure:: plots/PR.png  
   :name: figPRdrag

   Drift rate of the Poynting-Robertson drag. Computed after 10000 years and averaged over 1 orbit.



