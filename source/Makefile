SM=30
ARCH = -arch sm_$(SM)

# Build Data
HG_BRANCH := `hg branch`
HG_COMMIT := `hg id -i`
BUILD_DATE := "`date '+%F %T'`"
BUILD_SYSTEM := "`uname -a`"
BUILD_PATH := "`pwd`"
BUILD_DATA = -DHG_BRANCH=\"$(HG_BRANCH)\" -DHG_COMMIT=\"$(HG_COMMIT)\" -DBUILD_DATE=\"$(BUILD_DATE)\" -DBUILD_SYSTEM=\"$(BUILD_SYSTEM)\" -DBUILD_PATH=\"$(BUILD_PATH)\" -DBUILD_SM=\"$(SM)\"

source = genga.cu Host2.cu Orbit2.cu Energy.cu output.cu integrator.cu gas.cu
headers = define.h FG2.h BSSingle.h Encounter3.h Kick3.h Kick4.h HC.h BSB.h BSB64.h BSB128.h BSBsmall.h BSB64small.h BSBM.h ComEnergy.h directAcc.h 
genga: $(source) $(headers)
	nvcc $(ARCH) -o genga $(source) $(BUILD_DATA)
