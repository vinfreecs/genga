.. _Gas:

Gas disk
========

The gas disk is implemented according to :cite:p:`Morishima2010`. It supports gas drag, Type I migration and drag enhancement for small particles.
The parameters for the gas disk can be set with the following arguments:


In the :ref:`param.dat<ParamFile>` file

- :literal:`Use gas disk`
- :literal:`Use gas disk potential`
- :literal:`Use gas disk enhancement`
- :literal:`Use gas disk drag`
- :literal:`Use gas disk tidal dampening`
- :literal:`Gas dTau_diss`
- :literal:`Gas disk inner edge`:
- :literal:`Gas disk outer edge`:
- :literal:`Gas disk grid outer edge`:
- :literal:`Gas disk grid dr`:
- :literal:`Gas Sigma_10`
- :literal:`Gas alpha`
- :literal:`Gas beta`
- :literal:`Gas Mgiant`
- :literal:`Gas file name`

In the :ref:`define.h<Define>` file

- :literal:`def_Gasnz_g`: Number of cells in z direction for gas grid
- :literal:`def_Gasnz_p`: Number of cells in z direction for particle grid
- :literal:`def_h_1`: scale height at 1AU for c = 1km/s
- :literal:`def_M_Enhance`: factor for enhancement
- :literal:`def_Mass_pl`: factor for enhancement
- :literal:`def_fMass_min`: factor for enhancement
- :literal:`def_Gas_cd`: numerical gas drag coefficient


Gas disk structure
------------------

The gas disk structure is implemented as a uniform disk in space, which decays exponentially in time (:cite:p:`Morishima2010`).

.. math::
   :label: eq_Gas1

   \Sigma_{gas}(r,t) = \Sigma{gas,0} \left( \frac{r}{1 AU} \right)^{-\alpha} \exp \left( - \frac{t}{\tau_{decay}} \right),

with the gas surface density at 1 AU :math:`\Sigma{gas,0}`, the dissipation time :math:`\tau_{decay}` and the power law exponent :math:`\alpha`. 

The scale height of the gas disk is set to

.. math::
   :label: eq_Gas2

   h(r) = h_0 \frac{r}{1 AU} \left( \frac{r}{1 AU} \right)^{\beta},

with the scale height at 1AU :math:`h_0`.


Gas disk physical range in r
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The range of the gas disk can be set with the :literal:`Gas disk inner edge`: and :literal:`Gas disk outer edge`: parameters.
Outside of these boundaries, the gas disk density is 0.


Gas disk grid
^^^^^^^^^^^^^

In order to calculate the gas disk gravitational effect on the particles, the gas disk gravitational force in r and z is tabulated 
and stored in a gas disk grid. While the tabulated values of the grid respect the entire gas disk, ranging from the inner edge to
the outer edge, the gas disk itself can have a smaller range in r. This is especially useful, when the gas disk extends a broader range
than the particles. Therefore, the outer range of the gas disk grid can be set by a different parameter :literal:`Gas disk grid outer edge`:.
The inner edge of the gas disk grid corresponds to the physical inner edge of the disk, :literal:`Gas disk inner edge`:

The spacing of the gas disk grid in r can be set with the parameter :literal:`Gas disk grid dr`.

When a particle is located outside of the gas disk grid, then the effect of the gas disk potential on the particles is not applied by using
the tabulated values, but with a simpler approach according to Ward(1981).



Gas drag force
--------------

The gas drag force is enabled with the :literal:`Use gas disk drag` parameter. The gas drag force is implemented as

.. math::
   :label: eq_Gas3

   \mathbf{F}_{drag} = - \frac{1}{2m} c_D \pi r^2 \rho_{gas} | \mathbf{v}_{rel} | \mathbf{v}_{rel},

with the radius of the particle :math:`r` and the mass of the particle :math:`m`.
The numerical coefficient :math:`c_D` is set to 2 (:cite:p:`Morishima2010`). The value of :math:`c_D` can be changed in the :ref:`define.h<Define>` file.




Gas disk file
-------------

When a gas disk file name is specified in the :literal:`Gas file name` parameter. Then the gas disk structure is red from this file.
The file must contain the following columns::

        time0 r Sigma h
        time1 r Sigma h
        .
        .
        .

with:

 - time in years.
 - r the distance from the cell to the star in AU.
 - Sigma, the surface density at the cell location r,in in g/:math:`\text{cm}^3`.
 - h, the gas disk scale height at the cell location r, in AU.  
