SM=30
ARCH = -arch sm_$(SM)
objects = genga.o Host2.o Orbit2.o Energy.o output.o gas.o integrator.o Kick3.o HC.o FG2.o BSSingle.o directAcc.o Encounter3.o BSB.o BSB64.o BSBsmall.o BSBM.o
genga: $(objects)
	nvcc -Xnvlink -v $(ARCH) -o genga $(objects)

genga.o: genga.cu define.h
	nvcc -Xptxas -v $(ARCH) -dc genga.cu
Host2.o: Host2.cu Host2.h define.h
	nvcc -Xptxas -v $(ARCH) -dc Host2.cu
Orbit2.o: Orbit2.cu Orbit2.h define.h
	nvcc -Xptxas -v $(ARCH) -dc Orbit2.cu
Energy.o: Energy.cu Energy.h
	nvcc -Xptxas -v $(ARCH) -dc Energy.cu
output.o: output.cu output.h define.h
	nvcc -Xptxas -v $(ARCH) -dc output.cu
gas.o: gas.cu gas.h define.h
	nvcc -Xptxas -v $(ARCH) -dc gas.cu
integrator.o: integrator.cu integrator.h define.h
	nvcc -Xptxas -v $(ARCH) -dc integrator.cu
Kick3.o: Kick3.cu Kick3.h
	nvcc -Xptxas -v $(ARCH) -dc Kick3.cu
HC.o: HC.cu HC.h
	nvcc -Xptxas -v $(ARCH) -dc HC.cu
FG2.o: FG2.cu FG2.h define.h
	nvcc -Xptxas -v $(ARCH) -dc FG2.cu
BSSingle.o: BSSingle.cu BSSingle.h
	nvcc -Xptxas -v $(ARCH) -dc BSSingle.cu
BSB.o: BSB.cu BSB.h
	nvcc -Xptxas -v $(ARCH) -dc BSB.cu
BSB64.o: BSB64.cu BSB64.h
	nvcc -Xptxas -v $(ARCH) -dc BSB64.cu
BSBsmall.o: BSBsmall.cu BSBsmall.h
	nvcc -Xptxas -v $(ARCH) -dc BSBsmall.cu
BSBM.o: BSBM.cu BSBM.h
	nvcc -Xptxas -v $(ARCH) -dc BSBM.cu
directAcc.o: directAcc.cu directAcc.h
	nvcc -Xptxas -v $(ARCH) -dc directAcc.cu
Encounter3.o: Encounter3.cu Encounter3.h
	nvcc -Xptxas -v $(ARCH) -dc Encounter3.cu

.PHONY: clean, AR
clean: 
	rm genga $(objects)
