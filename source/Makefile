SM=30
ARCH = -arch sm_$(SM)
source = genga.cu Host2.cu Orbit2.cu Energy.cu output.cu integrator.cu gas.cu
headers = define.h FG2.h BSSingle.h Encounter3.h Kick3.h BSB.h BSB64.h BSB128.h BSBsmall.h BSBM.h ComEnergy.h directAcc.h 
genga: $(source) $(headers)
	nvcc $(ARCH) -o genga $(source)
