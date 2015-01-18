OS = $(shell uname)
SM=30
ARCH = -arch sm_$(SM)
path = ../source/
source = gengaGL.cu $(path)Host2.cu $(path)Orbit2.cu $(path)Energy.cu $(path)output.cu $(path)integrator.cu $(path)gas.cu
headers = $(path)define.h $(path)FG2.h $(path)BSSingle.h $(path)Encounter3.h $(path)Kick3.h $(path)HC.h $(path)BSB.h $(path)BSB64.h $(path)BSB128.h $(path)BSBsmall.h $(path)BSB64small.h $(path)BSBM.h $(path)ComEnergy.h $(path)directAcc.h 

ifeq ($(OS), Linux)
GLUT = -lglut -lGLU
else
GLUT = -framework OpenGL -framework GLUT
endif
gengaGL: $(source) $(headers)
	nvcc $(ARCH) -o gengaGL $(source) $(GLUT)
