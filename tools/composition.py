'''
This script creates a list for all remaining bodies containing all ids of bodies it accreated.
The first column is the particle id (-2 and -3 for ejections at the inner and outher edge)
The second column is the particle id of the original body it has acreated

The name has to be set according to the simulation name


run with:
python3 composition -in <name>
e.g. python3 composition -in test

Author: Simon Grimm
Date:	April 2024

'''

import numpy as np
import argparse

def main(name):

	Collfile = 'Collisions%s.dat' % name
	Ejectfile = 'Ejections%s.dat' % name
	Outfile = 'Out%s_000000000000.dat' % name

	ID0 = np.loadtxt(Outfile, unpack=True, usecols=(1,), dtype=int)
	m0 = np.loadtxt(Outfile, unpack=True, usecols=(2,), dtype=float)


	IDi, IDj = np.loadtxt(Collfile, unpack=True, usecols=(1, 13), dtype=int)
	mi, mj = np.loadtxt(Collfile, unpack=True, usecols=(2, 14), dtype=float)

	maxID = np.max(ID0)

	#print(maxID)

	source = np.zeros(maxID + 1)
	ii = np.zeros(maxID + 1)


	for i in range(len(source)):
		source[i] = i
		ii[i] = i


	for i in range(len(IDi) -1, -1, -1):
		if(mj[i] < mi[i]):
			source[IDj[i]] = source[IDi[i]]
		if(mi[i] < mj[i]):
			source[IDi[i]] = source[IDj[i]]

		if(mi[i] == mj[i]):
			if(IDj[i] < IDi[i]):
				source[IDi[i]] = source[IDj[i]]
			if(IDi[i] < IDj[i]):
				source[IDj[i]] = source[IDi[i]]

	ID, Eject = np.loadtxt(Ejectfile, unpack=True, usecols=(1, 13), dtype=int)



	for i in range(len(Eject)):
		if(Eject[i] == -2):
			for j in range(len(source)):
				if(source[j] == ID[i]):
					source[j] = -2
		if(Eject[i] == -3):
			for j in range(len(source)):
				if(source[j] == ID[i]):
					source[j] = -3

	#sort the list
	s_sorted = sorted(zip(source, ii))

	a, b = zip(*s_sorted)

	IDl = ID0.tolist()

	for i in range(len(source)):

		#print(i, int(source[i])
		try:
			ii = IDl.index(int(b[i]))
			print(int(a[i]), int(b[i]))
		except:
			ii = -1


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('-in', '--input', type=str,
	help='input name', default = 'test')

	args = parser.parse_args()

	name = args.input
	if(name == ''):
		print("Error, no input name specified, run python3 composition.py -in <name>")



	main(name)
	 
