SM=30
ARCH = -arch sm_$(SM)

# Build Data
GIT_BRANCH := `git rev-parse --abbrev-ref HEAD`
GIT_COMMIT := `git rev-parse HEAD`
BUILD_DATE := "`date`"
BUILD_SYSTEM := "`uname -a`"
BUILD_PATH := "`pwd`"
BUILD_DATA = -DGIT_BRANCH=\"$(GIT_BRANCH)\" -DGIT_COMMIT=\"$(GIT_COMMIT)\" -DBUILD_DATE=\"$(BUILD_DATE)\" -DBUILD_SYSTEM=\"$(BUILD_SYSTEM)\" -DBUILD_PATH=\"$(BUILD_PATH)\" -DBUILD_SM=\"$(SM)\"

source = genga.cu Host2.cu Orbit2.cu Energy.cu output.cu integrator.cu gas.cu 
headers = define.h Host2.h Orbit2.h FG2.h BSSingle.h Encounter3.h Kick3.h Kick4.h HC.h BSB.h BSBM.h BSB64M.h BSA.h ComEnergy.h directAcc.h force.h forceYarkovskyOld.h TTVStep2.h BSTTV.h  BSRV.h Scan.h
genga: $(source) $(headers)
	nvcc $(ARCH) --compiler-options -Wall -o genga $(source) $(BUILD_DATA)
#	nvcc $(ARCH) --ptxas-options=-v --compiler-options -Wall -o genga $(source) $(BUILD_DATA)
#	nvcc $(ARCH) -ccbin=g++-7 --compiler-options -Wall -o genga $(source) $(BUILD_DATA)
#	nvcc $(ARCH) -rdc=true -o genga $(source) $(BUILD_DATA)
