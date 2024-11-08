import numpy as np
import os


# This script can be used to increase the collision precision of a existing collisions file
# This is done by redoing a single time step containing only the two involved collision partners



outfilename = "initial_correct.dat"


#find simulation name in param.dat file
with open("param.dat") as pfile:
	for line in pfile:
		if("Output name =" in line):
			name = line.split(" = ")[1]
			name = name.split()[0]
			print("|",name,"|")
		if("Time step in days =" in line):
			dt = line.split(" = ")[1]
			dt = dt.split()[0]
			dt = float(dt)
			print("|",dt,"|")

name2 = name + "_correct"
name2T = name + "_correctT"

filename = "Collisions%s.dat" % name
filename2 = "Collisions%s.dat" % name2T

print(name, name2)

os.system("rm Collisions%s.dat" % name2)

count = 0
with open(filename) as file:
	for line in file:

		t = line.split()[0]
		i1 = line.split()[1]
		m1 = line.split()[2]
		r1 = line.split()[3]
		x1 = line.split()[4]
		y1 = line.split()[5]
		z1 = line.split()[6]
		vx1 = line.split()[7]
		vy1 = line.split()[8]
		vz1 = line.split()[9]
		Sx1 = line.split()[10]
		Sy1 = line.split()[11]
		Sz1 = line.split()[12]
		i2 = line.split()[13]
		m2 = line.split()[14]
		r2 = line.split()[15]
		x2 = line.split()[16]
		y2 = line.split()[17]
		z2 = line.split()[18]
		vx2 = line.split()[19]
		vy2 = line.split()[20]
		vz2 = line.split()[21]
		Sx2 = line.split()[22]
		Sy2 = line.split()[23]
		Sz2 = line.split()[24]

		outfile = open(outfilename, "w")
		print(t, i1, m1, r1, x1, y1, z1, vx1, vy1, vz1, Sx1, Sy1, Sz1, file = outfile)
		print(t, i2, m2, r2, x2, y2, z2, vx2, vy2, vz2, Sx2, Sy2, Sz2, file = outfile)
		outfile.close()

		#Do Backward integration to move the particles in a non-overlapping situation
		os.system("./genga -dt %g -in %s -out %s" % (-dt, outfilename, name2T))

		with open(filename2) as file:
			for line in file:

				t = line.split()[0]
				i1 = line.split()[1]
				m1 = line.split()[2]
				r1 = line.split()[3]
				x1 = line.split()[4]
				y1 = line.split()[5]
				z1 = line.split()[6]
				vx1 = line.split()[7]
				vy1 = line.split()[8]
				vz1 = line.split()[9]
				Sx1 = line.split()[10]
				Sy1 = line.split()[11]
				Sz1 = line.split()[12]
				i2 = line.split()[13]
				m2 = line.split()[14]
				r2 = line.split()[15]
				x2 = line.split()[16]
				y2 = line.split()[17]
				z2 = line.split()[18]
				vx2 = line.split()[19]
				vy2 = line.split()[20]
				vz2 = line.split()[21]
				Sx2 = line.split()[22]
				Sy2 = line.split()[23]
				Sz2 = line.split()[24]

				outfile = open(outfilename, "w")
				print(t, i1, m1, r1, x1, y1, z1, vx1, vy1, vz1, Sx1, Sy1, Sz1, file = outfile)
				print(t, i2, m2, r2, x2, y2, z2, vx2, vy2, vz2, Sx2, Sy2, Sz2, file = outfile)
				outfile.close()

		#Do Forward integration to collide particles again with a higher precision
		os.system("./genga -dt %g -in %s -out %s -collPrec %g" % (dt, outfilename, name2T, 1.0e-7))

		os.system("cat Collisions%s.dat >> Collisions%s.dat" % (name2T, name2))

		count += 1

		#if(count == 1):
		#	break


