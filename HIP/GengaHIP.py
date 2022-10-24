

GengaPath = '../source/'

#set def_OldShuffle to 1 
#set USE_RANDOM to 0

########################################
#Makefile
########################################

filename1 = '%s/Makefile' % GengaPath
filename2 = 'Makefile'

file1 = open(filename1, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for i in range(len(Lines)):

    line = Lines[i]

    if(line.find('.cu') != -1):
        line = line.replace('.cu', '.cpp')
    if(line.find('nvcc') != -1):
        line = line.replace('nvcc', 'hipcc')
    if(line.find('--compiler-options') != -1):
        line = line.replace('--compiler-options', '')

    print(line, file=file2, end='')

file1.close()
file2.close()



########################################
#define.h
########################################

filename1 = '%s/define.h' % GengaPath
filename2 = 'define.h'

file1 = open(filename1, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for i in range(len(Lines)):

    line = Lines[i]
    
    if(line.find('#define def_OldShuffle') != -1):
        print('#include "hip/hip_runtime.h"', file=file2)


    if(line.find('#define def_OldShuffle') != -1):
        line = line.replace('0', '1')
    if(line.find('define USE_RANDOM') != -1):
        line = line.replace('1', '0')

    #if(line.find('#include <curand_kernel.h>') != -1):  #<----------------- fix that
    #    line = line.replace('#include', '//#include')

    if(line.find('curand') != -1):
        line = line.replace('curand', 'hiprand')

    print(line, file=file2, end='')

file1.close()
file2.close()


########################################
# *.cu
########################################

files = ['genga', 'Host2', 'Orbit2', 'Energy', 'output', 'integrator', 'gas']

#not used: naf2.cu
for fi in files:

    filename1 = '%s/%s.cu' % (GengaPath, fi)
    filename2 = '%s.cpp' % fi

    file1 = open(filename1, 'r')
    file2 = open(filename2, 'w')

    Lines = file1.readlines()

    for i in range(len(Lines)):

        line = Lines[i]
      
        if(line.find('cudaDeviceProp') != -1):
            line = line.replace('cudaDeviceProp', 'hipDeviceProp_t')

        if(line.find('devProp.deviceOverlap') != -1):
            line = line.replace('devProp.deviceOverlap', '0')
     
        if(line.find('cudaHostAlloc') != -1):
            line = line.replace('cudaHostAlloc', 'hipHostMalloc')
    
        if(line.find('cudaFreeHost') != -1):
            line = line.replace('cudaFreeHost', 'hipHostFree')

        if(line.find('cudaHostAllocMapped') != -1):
            line = line.replace('cudaHostAllocMapped', 'hipHostMallocMapped')
     
        if(line.find('cudaHostAllocDefault') != -1):
            line = line.replace('cudaHostAllocDefault', 'hipHostMallocDefault')
  
        if(line.find('curand') != -1):
            line = line.replace('curand', 'hiprand')
      
        if(line.find('cuda') != -1):
            line = line.replace('cuda', 'hip')

        print(line, file=file2, end='')

    file1.close()
    file2.close()

########################################
# *.h
########################################

files = ['BSA', 'BSB64M', 'BSB', 'BSBM', 'BSRV', 'BSSingle', 'BSTTV', 'ComEnergy', 'directAcc', 'Encounter3', 
        'FG2', 'force', 'forceYarkovskyOld', 'HC', 'Host2', 'Kick3', 'Kick4', 'Orbit2', 'Rcrit', 'Scan',
        'TTVAll', 'TTVStep2', 'createparticles', 'convert', 'bvh']

#not used: naf.h

for fi in files:

    filename1 = '%s/%s.h' % (GengaPath, fi)
    filename2 = '%s.h' % fi

    file1 = open(filename1, 'r')
    file2 = open(filename2, 'w')

    Lines = file1.readlines()

    for i in range(len(Lines)):

        line = Lines[i]

        if(fi == 'BSA'):
            if(line.find('__shared__ volatile double4') != -1):
                print(fi, 'remove volatile')
                line = line.replace('volatile', '')

        if(line.find('curand') != -1):
            line = line.replace('curand', 'hiprand')

        if(line.find('cuda') != -1):
            line = line.replace('cuda', 'hip')

        print(line, file=file2, end='')

    file1.close()
    file2.close()

print('Done')
