import matplotlib
#matplotlib.use('PS')
matplotlib.use('agg')

import pylab as pl
import numpy as np

pl.rc('font', size=12)
params = {'legend.fontsize': 12}
pl.rcParams.update(params)


#yt = np.arange(1.0)*0.1


scale = 2.0e4
step = 1000

pl.figure(figsize=(8, 6))

for jj in range(0 * step, 10000 * step, step):

	filename = 'aeir10e2G_%.12d.dat' % jj
	
	print(filename)

	ax1=pl.subplot(111)
	t, i, a, e, inc, Omega, w, T, E, M, m, r = np.loadtxt(filename, unpack=True)

	r0 = r[0]
	m = m + 1.0 / scale

	m[0] = r0

	
	pl.scatter(a, e, c = i, edgecolors = '', s= m * scale)
	pl.ylim(0,0.2)
	pl.xlim(0,8)
	pl.xlabel('a [AU]')
	pl.ylabel('e')
	#pl.yticks(yt)

#	ax1.set_aspect('equal')
	time = 't = %g yr' % t[0]
	pl.text(7.5, 0.19, time, ha='right',va='center')

	name = 'plotaei%.5d.png' % (jj / step)
	pl.savefig(name, dpi=300)
	pl.clf()
	

