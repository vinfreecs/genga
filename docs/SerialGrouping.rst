.. _SerialGrouping:
 
Exact reproducible results
==========================

Usually, the outcome of a parallel numerical calculation is not exactly reproducible. The reason is,
that calculations are done with a finite precision and that the outcome depends on the order of the operations.
In general, we have :math:`a + b + c \neq a + c + b`. And since the planetary N-body problem is chaotic, already
tiny differences in the calculation can lead to a different result on individual bodies. Even the number of
formed planets can vary from simulation to simulation. A detailed study about this effect is given in
:cite:p:`Hoffmann2017`.

However it is possible to force GENGA to exactly reproduce a given outcome. This is possible because most
parallel operations are using parallel reduction sums with a fixed order. The only place which has not a
fixed order, is the creation of the close encounter pair lists. By introducing an additional sorting step
on the close encounter pairs lists, also this order can be fixed  and the result of a given initial condition
is always the same. It is important to note that this does not mean that the results are 'true', they still
suffer from small round-off errors, just this error is always the same. Since the additional sorting step
introduces also a performance penalty, it is not recommended to use this mode of GENGA for production runs.
But it can be very useful to check if a GPU works correctly and it allows also to eliminate memory leaks in
a code. 

The enable the exact reproducible outcome mode on GENGA, the :literal:`Serial Grouping` option in the :ref:`param.dat<ParamFile>` file must be set to 1.

