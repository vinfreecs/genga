# This scripts cleans the energy file, when the simulations was interrrupted between
# output time intervals. For example when a wall time of 24 hours was used.

# run with:
# python3 cleanEnergy.py -n test -ei 100 -dt 6.0
# -n is the name of the simulation output
# -ei is the energy output interval of the simulation
# -dt is the time step

# Date: Okt 2025
# Author: Simon Grimm

import numpy as np
import math
import argparse




def main(name, T, dt):

	filename = "Energy%s.dat" % name
	outfile = open(("EnergyClean%s.dat" % name), "w")

	print(filename)

	efile = open(filename, "r")

	i = 0
	for line in efile.readlines():
		t = np.float64(line.split(" ")[0])
		tt = "%.16g" % (t)
		
		et = "%.16g" % (i * T * dt / 365.25)
		if(tt == et):
			i += 1

			#print(tt, et, tt == et)
			print(line, end=' ', file = outfile)

	outfile.close()
	efile.close()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('-n', '--name', type=str, help='Name', default = 'test')
	parser.add_argument('-ei', '--ei', type=int, help='energy output interval', default = 100)
	parser.add_argument('-dt', '--dt', type=np.float64, help='time step', default = 100)


	args = parser.parse_args()

	main(args.name, args.ei, args.dt)
