SM=35
ARCH = -arch sm_$(SM)
source = KE.cu
headers = 
KE: $(source) $(headers)
	nvcc $(ARCH) -o KE $(source)

