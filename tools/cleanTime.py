# This scripts cleans the time file, when the simulations was interrrupted between
# output time intervals. For example when a wall time of 24 hours was used.

# run with:
# python3 cleanTime.py -n FP32High -ci 730500
# -n is the name of the simulation output
# -ci is the coordinate output interval of the simulation

# Date: Jan 2023
#Author: Simon Grimm

import numpy as np
import math
import argparse



#name = 'FP32High'

#T = 730500


def main(name, T):

	filename = "time%s.dat" % name
	outfile = open(("timeClean%s.dat" % name), "w")

	print(filename)

	s, t = np.loadtxt(filename, unpack=True)

	tt = 0.0


	for i in range(len(s)):
		tt += t[i]
		
		if(i > 0 and s[i] == s[i-1]):
			break

		if(s[i] % T == 0):
			print(s[i], tt, file=outfile)
			tt = 0.0 

	outfile.close()

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('-n', '--name', type=str, help='Name', default = 'test')
	parser.add_argument('-ci', '--ci', type=int, help='output interval', default = 100)


	args = parser.parse_args()

	main(args.name, args.ci)
