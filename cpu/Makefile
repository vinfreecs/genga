
# Build Data
GIT_BRANCH := `git rev-parse --abbrev-ref HEAD`
GIT_COMMIT := `git rev-parse HEAD`
BUILD_DATE := "`date`"
BUILD_SYSTEM := "`uname -a`"
BUILD_PATH := "`pwd`"
BUILD_DATA = -DGIT_BRANCH=\"$(GIT_BRANCH)\" -DGIT_COMMIT=\"$(GIT_COMMIT)\" -DBUILD_DATE=\"$(BUILD_DATE)\" -DBUILD_SYSTEM=\"$(BUILD_SYSTEM)\" -DBUILD_PATH=\"$(BUILD_PATH)\"

source = gengaCPU.cu Host2CPU.cu Orbit2CPU.cu EnergyCPU.cu outputCPU.cu integratorCPU.cu gasCPU.cu

headers = define.h Host2CPU.h Orbit2CPU.h FG2CPU.h BSSingleCPU.h Encounter3CPU.h Kick3CPU.h Kick4CPU.h HCCPU.h BSBCPU.h BSACPU.h ComEnergyCPU.h directAccCPU.h forceCPU.h forceYarkovskyOldCPU.h ScanCPU.h RcritCPU.h convertCPU.h createparticlesCPU.h bvhCPU.h 

gengaCPU: $(source) $(headers)
	# with gprof
	#nvcc -Xcompiler=-fopenmp --compiler-options -Wall -O3 -pg -o gengaCPU $(source) $(BUILD_DATA)
	nvcc -Xcompiler=-fopenmp --compiler-options -Wall -O3 -o gengaCPU $(source) $(BUILD_DATA)
