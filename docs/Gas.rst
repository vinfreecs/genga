.. _Gas:

Gas disk
===========================
The gas disk is implemented according to :cite:p:`Morishima2010`. It supports gas drag, Type I migration and drag enhancement for small particles.
The parameters for the gas disk can be set with the following arguments:


In the :ref:`param.dat<ParamFile>` file

- :literal:`Use gas disk`
- :literal:`Use gas disk potential`
- :literal:`Use gas disk enhancement`
- :literal:`Use gas disk drag`
- :literal:`Use gas disk tidal dampening`
- :literal:`Gas dTau_diss`
- :literal:`Gas Sigma_10`
- :literal:`Gas alpha`
- :literal:`Gas beta`
- :literal:`Gas Mgiant`
- :literal:`Gas file name`

In the :ref:`define.h<Define>` file

- :literal:`def_Gasnr_g`: Number of cells in r direction for gas grid
- :literal:`def_Gasnz_g`: Number of cells in z direction for gas grid
- :literal:`def_Gasnr_p`: Number of cells in r direction for particle grid
- :literal:`def_Gasnz_p`: Number of cells in z direction for particle grid
- :literal:`def_h_1`: scale height at 1AU for c = 1km/s
- :literal:`def_M_Enhance`: factor for enhancement
- :literal:`def_Mass_pl`: factor for enhancement
- :literal:`def_fMass_min`: factor for enhancement

Gas disk file
-------------

When a gas disk file name is specified in the :literal:`Gas file name` parameter. Then the gas disk structure is read from this file.
The file must contain the following columns::

        time0 r Sigma h
        time1 r Sigma h
        .
        .
        .

with:

 - time in years.
 - r the distance from the cell to the star in AU.
 - Sigma, the surface desity at the cell location r,in in g/:math:`\text{cm}^3`.
 - h, the gas disk scale height at the cell location r, in AU.  
