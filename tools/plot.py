import matplotlib
#matplotlib.use('PS')

import pylab as pl
import numpy as np

pl.rc('font', size=12)
params = {'legend.fontsize': 12}
pl.rcParams.update(params)

ymax = 7
xmax = 7

yt = np.arange(1.0)*0.1

step = 100

scale = 4e10

pl.figure(figsize=(8, 6))

for jj in range(0 * step, 1000, step):

	filename = 'Outt_%.12d.dat' % jj
	
	print filename

	ax1=pl.subplot(111)
	t, i, m ,r, x, y, z, vx, vy, vz, Sx, Sy, Sz, t1, t2, t3, t4, t5, t6, t7, t8 = np.loadtxt(filename, unpack=True)
	
	pl.scatter(x, y, c = 'red', edgecolors='', s= 2)#m * scale)
	pl.ylim(-ymax,ymax)
	pl.xlim(-xmax,xmax)
	pl.xlabel('x [AU]')
	pl.ylabel('y [AU]')
	#pl.yticks(yt)

#	pl.axis('equal')
	ax1.set_aspect('equal')
	time = 't = %g yr' % t[0]
	pl.text(6.5, 6.7, time, ha='right',va='center')

	name = 'plot%.5d.png' % (jj / step)
	pl.savefig(name, dpi=300)
	pl.clf()
	

