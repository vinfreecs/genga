.. _RotationalDeformation:

Rotational Deformation
======================

The rotational deformation force can be enabled with the :literal:`Use Rotational Deformation` parameter in the :literal:`param.dat` file.


Parameters for the tidal forces:

 - :literal:`Star fluid Love Number`
 - :literal:`Star spin_x`
 - :literal:`Star spin_y`
 - :literal:`Star spin_z`
 - :literal:`Star Ic`

and the initial conditions:

 - k2f
 - Sx
 - Sy
 - Sz
 - Ic

When a viscous body is rotating, then its shape is transformed into a symmetric oblate ellipsoid.
The potential energy of a system containing two oblate bodies 0 and 1 is give in :cite:p:`Moyer1971`, Equation 158, :cite:p:`Moyer2003`
or :cite:p:`Correia2011` as

.. math::

    U = -\frac{Gm_0 m_1}{r} \left[ 1 - \sum_{i=0,1} \sum_{n=1}^{\infty} J_n \left( \frac{R_i}{r} \right)^n Pn(\hat{r} \cdot \hat{\Omega_i}) \right],

with the mass :math:`m`, the physical radius :math:`R`, the distance between the bodies :math:`r`, the rotational angular velocity
:math:`\Omega` and the Legendre polynomial of degree :math:`n`, :math:`Pn(x)`. 

Further, we truncate the order :math:`n` to two and introduce the :math:`J_2` parameter :cite:p:`Correia2011`, :cite:p:`Bolmont2015`

.. math::

    J_{2,i} = k_{2f,i} \frac{\Omega^2_i R^3_i}{3G m_i}

and

.. math:: 

   J_{2,\star} = k_{2f,\star} \frac{\Omega^2_\star R^3_\star}{3G m_\star},

with :math:`k_{2f,i}` the second potential Love number (fluid Love number).

We follow the description in :cite:p:`Bolmont2015` and define the following quantities
(In :cite:p:`Bolmont2015`, :math:`C_\star` and :math:`C_i` are reversed):

.. math::

    C_{\star} = \frac{1}{2} G m_i m_\star J_{2,\star} R^2_\star

and

.. math:: 

    C_{i} = \frac{1}{2} G m_i m_\star J_{2,i} R^2_i.

Then the force due to the rotational deformation is given as :cite:p:`Bolmont2015`:

.. math::

   \mathbf{F_r} = \left\{ - \frac{3}{r^5_i} \left( C_\star + C_i\right) \right. \\ \nonumber
   \left. + \frac{15}{r^7_i} \left[ C_\star \frac{(\mathbf{r}_i \cdot \mathbf{\Omega}_\star)^2}{\mathbf{\Omega}_\star^2} +  C_i \frac{(\mathbf{r}_i \cdot \mathbf{\Omega}_i)^2}{\mathbf{\Omega}_i^2} \right] \right\} \mathbf{r}_i \\ \nonumber
    - \frac{6}{r^5_i} \left( C_\star \frac{\mathbf{r}_i \cdot \mathbf{\Omega}_\star}{\mathbf{\Omega}_\star^2} \mathbf{\Omega}_\star
    + C_i \frac{\mathbf{r}_i \cdot \mathbf{\Omega}_i}{\mathbf{\Omega}_i^2} \mathbf{\Omega}_i\right).
