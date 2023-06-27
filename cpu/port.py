import shutil
#shutil.copyfile('../source/output.cu', filename)

#manually Makefile
#manually Energy.cu
#manually HC.h
#manually ComEnergy.h
#manually BSB.h
#manually BSA.h


def kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2):
	if(line.find('_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
	if(line.find('threadIdx.x') != -1):
		line = line.replace('threadIdx.x', '0')
	if(line.find('blockIdx.x') != -1):
		line = line.replace('blockIdx.x', '0')
	if(line.find('blockDim.x') != -1):
		line = line.replace('blockDim.x', '1')
	if(line.find('threadIdx.y') != -1):
		line = line.replace('threadIdx.y', '0')
	if(line.find('blockIdx.y') != -1):
		line = line.replace('blockIdx.y', '0')
	if(line.find('blockDim.y') != -1):
		line = line.replace('blockDim.y', '1')
	if(line.find('__global__') != -1):
		line = line.replace('__global__ ', '')

	if(line.find('int %s =' % loop_id) != -1):
		i_id = line.split('int ')[1]
		i_id = i_id.split(';')[0]

	if(line.find('int %s =' % loop_id2) != -1):
		i_id2 = line.split('int ')[1]
		i_id2 = i_id2.split(';')[0]

	#insert loops
	i_if = 'if(%s < %s){' % (loop_id, loop_N)
	i_for = 'for(%s; %s < %s; ++%s){' % (i_id, loop_id, loop_N, loop_id)

	if(omp == 1):
		tab = line.split('i')[0]
		i_for = '#pragma omp parallel for\n' + tab + i_for


	if(line.find(i_if) != -1):
		line = line.replace(i_if, i_for)

	i_if = 'if(%s < %s){' % (loop_id2, loop_N2)
	i_for = 'for(%s; %s < %s; ++%s){' % (i_id2, loop_id2, loop_N2, loop_id2)
	if(line.find(i_if) != -1):
		line = line.replace(i_if, i_for)

	return line, i_id, i_id2


#This function replaces '_d' with '_h'
def DtoH(line):
	if(line.find('x4_d') != -1):
		line = line.replace('x4_d', 'x4_h')
	if(line.find('v4_d') != -1):
		line = line.replace('v4_d', 'v4_h')
	if(line.find('x4b_d') != -1):
		line = line.replace('x4b_d', 'x4b_h')
	if(line.find('v4b_d') != -1):
		line = line.replace('v4b_d', 'v4b_h')
	if(line.find('x4bb_d') != -1):
		line = line.replace('x4bb_d', 'x4bb_h')
	if(line.find('v4bb_d') != -1):
		line = line.replace('v4bb_d', 'v4bb_h')
	if(line.find('xold_d') != -1):
		line = line.replace('xold_d', 'xold_h')
	if(line.find('vold_d') != -1):
		line = line.replace('vold_d', 'vold_h')
	if(line.find('a_d') != -1):
		line = line.replace('a_d', 'a_h')
	if(line.find('ab_d') != -1):
		line = line.replace('ab_d', 'ab_h')
	if(line.find('abb_d') != -1):
		line = line.replace('abb_d', 'abb_h')
	if(line.find('N_d') != -1):
		line = line.replace('N_d', 'N_h')
	if(line.find('Nsmall_d') != -1):
		line = line.replace('Nsmall_d', 'Nsmall_h')
	if(line.find('index_d') != -1):
		line = line.replace('index_d', 'index_h')
	if(line.find('indexb_d') != -1):
		line = line.replace('indexb_d', 'indexb_h')
	if(line.find('indexbb_d') != -1):
		line = line.replace('indexbb_d', 'indexbb_h')
	if(line.find('spin_d') != -1):
		line = line.replace('spin_d', 'spin_h')
	if(line.find('spinb_d') != -1):
		line = line.replace('spinb_d', 'spinb_h')
	if(line.find('spinbb_d') != -1):
		line = line.replace('spinbb_d', 'spinbb_h')
	if(line.find('love_d') != -1):
		line = line.replace('love_d', 'love_h')
	if(line.find('Energy_d') != -1):
		line = line.replace('Energy_d', 'Energy_h')
	if(line.find('test_d') != -1):
		line = line.replace('test_d', 'test_h')
	if(line.find('rcrit_d') != -1):
		line = line.replace('rcrit_d', 'rcrit_h')
	if(line.find('rcritb_d') != -1):
		line = line.replace('rcritb_d', 'rcritb_h')
	if(line.find('rcritbb_d') != -1):
		line = line.replace('rcritbb_d', 'rcritbb_h')
	if(line.find('rcritv_d') != -1):
		line = line.replace('rcritv_d', 'rcritv_h')
	if(line.find('rcritvb_d') != -1):
		line = line.replace('rcritvb_d', 'rcritvb_h')
	if(line.find('rcritvbb_d') != -1):
		line = line.replace('rcritvbb_d', 'rcritvbb_h')
	if(line.find('aelimits_d') != -1):
		line = line.replace('aelimits_d', 'aelimits_h')
	if(line.find('aecount_d') != -1):
		line = line.replace('aecount_d', 'aecount_h')
	if(line.find('aecountT_d') != -1):
		line = line.replace('aecountT_d', 'aecountT_h')
	if(line.find('enccount_d') != -1):
		line = line.replace('enccount_d', 'enccount_h')
	if(line.find('enccountT_d') != -1):
		line = line.replace('enccountT_d', 'enccountT_h')
	if(line.find('K_d') != -1):
		line = line.replace('K_d', 'K_h')
	if(line.find('Kold_d') != -1):
		line = line.replace('Kold_d', 'Kold_h')
	if(line.find('StopTime_d') != -1):
		line = line.replace('StopTime_d', 'StopTime_h')
	if(line.find('naf.x_d') != -1):
		line = line.replace('naf.x_d', 'naf.x_h')
	if(line.find('naf.y_d') != -1):
		line = line.replace('naf.y_d', 'naf.y_h')
	if(line.find('nafx_d') != -1):
		line = line.replace('nafx_d', 'nafx_h')
	if(line.find('nafy_d') != -1):
		line = line.replace('nafy_d', 'nafy_h')
	if(line.find('Encpairs2_d') != -1):
		line = line.replace('Encpairs2_d', 'Encpairs2_h')
	if(line.find('Nencpairs2_d') != -1):
		line = line.replace('Nencpairs2_d', 'Nencpairs2_h')
	if(line.find('Encpairs_d') != -1):
		line = line.replace('Encpairs_d', 'Encpairs_h')
	if(line.find('Nencpairs_d') != -1):
		line = line.replace('Nencpairs_d', 'Nencpairs_h')
	if(line.find('Nencpairs3_d') != -1):
		line = line.replace('Nencpairs3_d', 'Nencpairs3_h')
	if(line.find('Encpairs3_d') != -1):
		line = line.replace('Encpairs3_d', 'Encpairs3_h')
	if(line.find('NBS_d') != -1):
		line = line.replace('NBS_d', 'NBS_h')
	if(line.find('U_d') != -1):
		line = line.replace('U_d', 'U_h')
	if(line.find('LI_d') != -1):
		line = line.replace('LI_d', 'LI_h')
	if(line.find('vcom_d') != -1):
		line = line.replace('vcom_d', 'vcom_h')
	if(line.find('coordinateBuffer_d') != -1):
		line = line.replace('coordinateBuffer_d', 'coordinateBuffer_h')
	if(line.find('coordinateBufferIrr_d') != -1):
		line = line.replace('coordinateBufferIrr_d', 'coordinateBufferIrr_h')
	if(line.find('time_d') != -1):
		line = line.replace('time_d', 'time_h')
	if(line.find('idt_d') != -1):
		line = line.replace('idt_d', 'idt_h')
	if(line.find('Energy0_d') != -1):
		line = line.replace('Energy0_d', 'Energy0_h')
	if(line.find('LI0_d') != -1):
		line = line.replace('LI0_d', 'LI0_h')
	if(line.find('Gridaecount_d') != -1):
		line = line.replace('Gridaecount_d', 'Gridaecount_h')
	if(line.find('Gridaicount_d') != -1):
		line = line.replace('Gridaicount_d', 'Gridaicount_h')
	if(line.find('PFlag_d') != -1):
		line = line.replace('PFlag_d', 'PFlag_h')
	if(line.find('acck_d') != -1):
		line = line.replace('acck_d', 'acck_h')
	if(line.find('EncFlag_d') != -1):
		line = line.replace('EncFlag_d', 'EncFlag_m')
	if(line.find('Encpairsb_d') != -1):
		line = line.replace('Encpairsb_d', 'Encpairsb_h')
	if(line.find('Ncoll_d') != -1):
		line = line.replace('Ncoll_d', 'Ncoll_m')
	if(line.find('Nenc_d') != -1):
		line = line.replace('Nenc_d', 'Nenc_m')
	if(line.find('scan_d') != -1):
		line = line.replace('scan_d', 'scan_h')
	if(line.find('EjectionFlag_d') != -1):
		line = line.replace('EjectionFlag_d', 'EjectionFlag_m')
	if(line.find('writeEnc_d') != -1):
		line = line.replace('writeEnc_d', 'writeEnc_h')
	if(line.find('NWriteEnc_d') != -1):
		line = line.replace('NWriteEnc_d', 'NWriteEnc_m')
	if(line.find('Coll_d') != -1):
		line = line.replace('Coll_d', 'Coll_h')
	if(line.find('Msun_d') != -1):
		line = line.replace('Msun_d', 'Msun_h')
	if(line.find('Spinsun_d') != -1):
		line = line.replace('Spinsun_d', 'Spinsun_h')
	if(line.find('Lovesun_d') != -1):
		line = line.replace('Lovesun_d', 'Lovesun_h')
	if(line.find('J2_d') != -1):
		line = line.replace('J2_d', 'J2_h')
	if(line.find('dt_d') != -1):
		line = line.replace('dt_d', 'dt_h')
	if(line.find('setElementsLine_d') != -1):
		line = line.replace('setElementsLine_d', 'setElementsLine_h')
	if(line.find('nFragments_d') != -1):
		line = line.replace('nFragments_d', 'nFragments_m')
	if(line.find('Fragments_d') != -1):
		line = line.replace('Fragments_d', 'Fragments_h')
	if(line.find('nFragments_d') != -1):
		line = line.replace('nFragments_d', 'nFragments_m')
	if(line.find('migration_d') != -1):
		line = line.replace('migration_d', 'migration_h')
	if(line.find('createFlag_d') != -1):
		line = line.replace('createFlag_d', 'createFlag_h')
	if(line.find('groupIndex_d') != -1):
		line = line.replace('groupIndex_d', 'groupIndex_h')
	if(line.find('random_d') != -1):
		line = line.replace('random_d', 'random_h')
	if(line.find('EnergySum_d') != -1):
		line = line.replace('EnergySum_d', 'EnergySum_h')

	if(line.find('morton_d') != -1):
		line = line.replace('morton_d', 'morton_h')
	if(line.find('sortRank_d') != -1):
		line = line.replace('sortRank_d', 'sortRank_h')
	if(line.find('sortCount_d') != -1):
		line = line.replace('sortCount_d', 'sortCount_h')
	if(line.find('sortIndex_d') != -1):
		line = line.replace('sortIndex_d', 'sortIndex_h')
	if(line.find('leafNodes_d') != -1):
		line = line.replace('leafNodes_d', 'leafNodes_h')
	if(line.find('internalNodes_d') != -1):
		line = line.replace('internalNodes_d', 'internalNodes_h')

	if(line.find('xt_d') != -1):
		line = line.replace('xt_d', 'xt_h')
	if(line.find('vt_d') != -1):
		line = line.replace('vt_d', 'vt_h')
	if(line.find('xp_d') != -1):
		line = line.replace('xp_d', 'xp_h')
	if(line.find('vp_d') != -1):
		line = line.replace('vp_d', 'vp_h')
	if(line.find('dx_d') != -1):
		line = line.replace('dx_d', 'dx_h')
	if(line.find('dv_d') != -1):
		line = line.replace('dv_d', 'dv_h')
	if(line.find('dt1_d') != -1):
		line = line.replace('dt1_d', 'dt1_h')
	if(line.find('t1_d') != -1):
		line = line.replace('t1_d', 't1_h')
	if(line.find('dtgr_d') != -1):
		line = line.replace('dtgr_d', 'dtgr_h')
	if(line.find('BSstop_d') != -1):
		line = line.replace('BSstop_d', 'BSstop_h')
	if(line.find('BSAstop_d') != -1):
		line = line.replace('BSAstop_d', 'BSAstop_h')
	if(line.find('n1_d') != -1):
		line = line.replace('n1_d', 'n1_h')
	if(line.find('n2_d') != -1):
		line = line.replace('n2_d', 'n2_h')

	if(line.find('__dmul_rn') != -1):
		line = line.replace('__dmul_rn', '')
		line = line.replace(',', ' *')
	if(line.find('__fmul_rn') != -1):
		line = line.replace('__fmul_rn', '')
		line = line.replace(',', ' *')
	if(line.find('__fma_rn') != -1):
		line = line.replace('__fma_rn', 'fma')

	if(line.find('cudaError_t') != -1):
		line = line.replace('cudaError_t', 'int')
	if(line.find('cudaEvent_t') != -1):
		line = line.replace('cudaEvent_t', 'timeval')
	if(line.find('cudaEventRecord(') != -1):
		line = line.replace('cudaEventRecord(', 'gettimeofday(&')


	if(line.find('cudaGetLastError()') != -1):
		line = line.replace('cudaGetLastError()', '0')
	if(line.find('cudaGetErrorString(error)') != -1):
		line = line.replace('cudaGetErrorString(error)', '"-"')
	if(line.find('cudaEventCreate') != -1):
		line = line.replace('cudaEventCreate', '//')
	if(line.find('cudaEventDestroy') != -1):
		line = line.replace('cudaEventDestroy', '//')
	if(line.find('cudaEventSynchronize') != -1):
		line = line.replace('cudaEventSynchronize', '//')
	if(line.find('cudaEventElapsedTime') != -1):
		line = line.replace('cudaEventElapsedTime', 'ElapsedTime')

	return line


def memorycopy(line):
	if(line.find('cudaMemcpy(') != -1):
		t0 = line.split('(')[1]
		t1 = t0.split(',')[0]
		t1 = t1.replace(' ', '')
		t2 = t0.split(',')[1]
		t2 = t2.replace(' ', '')

		#print(t1, t2, t1 == t2)
		if(t1 == t2):
			line = ''
		else:
			line = line.replace(', cudaMemcpyHostToDevice', '')
			line = line.replace(', cudaMemcpyDeviceToHost', '')
			line = line.replace('cudaMemcpy', 'memcpy')
	return line

addLoop = 0
i_id = ''
loop_id = ''
loop_N = ''
i_id2 = ''
loop_id2 = ''
loop_N2 = ''
omp = 0
remove = 0

################################################
# define.h 
################################################

filename = '../source/define.h'
filename2 = 'define.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:

	if(line.find('__constant__') != -1):
		line = line.replace('__constant__', 'extern')


	if(line.find('def_CPU 0') != -1):
		line = line.replace('def_CPU 0', 'def_CPU 1')

	if(line.find('math.h') != -1):
		print("#include <omp.h>", file=file2)

	print(line, file=file2, end='')

file1.close()
file2.close()

################################################
# genga.cu 
################################################

filename = '../source/genga.cu'
filename2 = 'gengaCPU.cu'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:

	if(line.find('Host2.h') != -1):
		line = line.replace('Host2.h', 'Host2CPU.h')
	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')
	if(line.find('TTVStep2.h') != -1):
		line = line.replace('TTVStep2.h', 'TTVStep2CPU.h')

	if(line.find('cudaSetDevice') != -1):
		continue 
	if(line.find('cudaDeviceSynchronize') != -1):
		continue 

	if(line.find('cudaMemset') != -1):
		line = line.replace('cudaMemset', 'memset')
		line = line.replace('_d', '_h')

	line = DtoH(line)
	line = memorycopy(line)

	print(line, file=file2, end='')

	# no MCMC for cpu version
	# replace cuda events

file1.close()
file2.close()


################################################
# integrator.cu 
################################################

filename = '../source/integrator.cu'
filename2 = 'integratorCPU.cu'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()


for line in Lines:
	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')
	if(line.find('Rcrit.h') != -1):
		line = line.replace('Rcrit.h', 'RcritCPU.h')
	if(line.find('Kick3.h') != -1):
		line = line.replace('Kick3.h', 'Kick3CPU.h')
	if(line.find('HC.h') != -1):
		line = line.replace('HC.h', 'HCCPU.h')
	if(line.find('FG2.h') != -1):
		line = line.replace('FG2.h', 'FG2CPU.h')
	if(line.find('Encounter3.h') != -1):
		line = line.replace('Encounter3.h', 'Encounter3CPU.h')
	if(line.find('BSB.h') != -1):
		line = line.replace('BSB.h', 'BSBCPU.h')
	if(line.find('BSBM.h') != -1):
		line = '//' + line
	if(line.find('BSBM3.h') != -1):
		line = '//' + line
	if(line.find('ComEnergy.h') != -1):
		line = line.replace('ComEnergy.h', 'ComEnergyCPU.h')
	if(line.find('convert.h') != -1):
		line = line.replace('convert.h', 'convertCPU.h')
	if(line.find('force.h') != -1):
		line = line.replace('force.h', 'forceCPU.h')
	if(line.find('forceYarkovskyOld.h') != -1):
		line = line.replace('forceYarkovskyOld.h', 'forceYarkovskyOldCPU.h')
	if(line.find('Kick4.h') != -1):
		line = line.replace('Kick4.h', 'Kick4CPU.h')
	if(line.find('BSA.h') != -1):
		line = line.replace('BSA.h', 'BSACPU.h')
	if(line.find('BSAM3.h') != -1):
		line = '//' + line
	if(line.find('Scan.h') != -1):
		line = line.replace('Scan.h', 'ScanCPU.h')
	if(line.find('createparticles.h') != -1):
		line = line.replace('createparticles.h', 'createparticlesCPU.h')
	if(line.find('bvh.h') != -1):
		line = line.replace('bvh.h', 'bvhCPU.h')


	#Remove functions
	if(line.find('int Data::tuneFG') != -1):
		remove = 1
	if(line.find('int Data::tuneRcrit') != -1):
		remove = 1
	if(line.find('int Data::tuneForce') != -1):
		remove = 1
	if(line.find('int Data::tuneKick') != -1):
		remove = 1
	if(line.find('int Data::tuneKickM3') != -1):
		remove = 1
	if(line.find('int Data::tuneHCM3') != -1):
		remove = 1
	if(line.find('int Data::tuneBVH') != -1):
		remove = 1
	if(line.find('int Data::tuneBS2') != -1):
		remove = 1

	if(line.find('void Data::firstKick_16') != -1):
		remove = 1
	if(line.find('void Data::firstKick_largeN') != -1):
		remove = 1
	if(line.find('void Data::firstKick_small(') != -1):
		remove = 1
	if(line.find('void Data::firstKick_M') != -1):
		remove = 1
	if(line.find('void Data::firstKick_M3') != -1):
		remove = 1
	if(line.find('void Data::BSBMCall') != -1):
		remove = 1
	if(line.find('void Data::BSBM3Call') != -1):
		remove = 1
	if(line.find('int Data::CollisionMCall') != -1):
		remove = 1
	if(line.find('int Data::bStepM') != -1):
		remove = 1
	if(line.find('int Data::bStepM3') != -1):
		remove = 1
	if(line.find('int Data::KickndevCall') != -1):
		remove = 1
	if(line.find('int Data::KickfirstndevCall') != -1):
		remove = 1
	if(line.find('int Data::step_1kernel') != -1):
		remove = 1
	if(line.find('int Data::step_16') != -1):
		remove = 1
	if(line.find('int Data::step_largeN') != -1):
		remove = 1
	if(line.find('int Data::step_small(') != -1):
		remove = 1
	if(line.find('int Data::step_M') != -1):
		remove = 1
	if(line.find('int Data::step_M3') != -1):
		remove = 1
	if(line.find('int Data::step_MSimple') != -1):
		remove = 1
	if(line.find('int Data::ttv_step') != -1):
		remove = 1

	######################

	if(remove == 1):
		if(line.startswith('}',0,1)):
			remove = 0
			continue

	if(remove == 1):
		continue

	
	if(line.find('__host__') != -1):
		line = line.replace('__host__ ', '')

	if(line.find('cudaDeviceSynchronize') != -1):
		continue

	if(line.find('cudaMemset') != -1):
		line = line.replace('cudaMemset', 'memset')
		line = line.replace('_d', '_h')

	if(line.find('cudaFree') != -1):
		continue 

	#Add loops
	if(line.find('void initialb_kernel') != -1):
		loop_id = 'id'
		loop_N = 'NBNencT'
		addLoop = 1

	if(line.find('void test_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void save_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void testA_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0


	if(line.find('initialb_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('test_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('save_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('PoincareSection_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('setEnc3_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('groupS2_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('SortSb_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('RcritS_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('kickS_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('fgS_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('encounter_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('EncpairsZeroC_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('Rcrit_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('acc4C_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('kick32Ab_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('fg_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('setNencpairs_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	line = DtoH(line)

	line = memorycopy(line)

	print(line, file=file2, end='')

file1.close()
file2.close()



################################################
# Host2.h 
################################################

filename = '../source/Host2.h'
filename2 = 'Host2CPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:

	print(line, file=file2, end='')

file1.close()
file2.close()


################################################
# Host2.cu 
################################################

filename = '../source/Host2.cu'
filename2 = 'Host2CPU.cu'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:

	if(line.find('Host2.h') != -1):
		line = line.replace('Host2.h', 'Host2CPU.h')


	if(line.find('cudaMemcpy') != -1):
		continue 
	if(line.find('cudaMalloc') != -1):
		continue 
	if(line.find('cudaMemset') != -1):
		continue 
	if(line.find('cudaFree') != -1):
		continue 
	line = DtoH(line)

	print(line, file=file2, end='')

file1.close()
file2.close()

################################################
# Orbit2.h 
################################################

filename = '../source/Orbit2.h'
filename2 = 'Orbit2CPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:
	if(line.find('Host2.h') != -1):
		line = line.replace('Host2.h', 'Host2CPU.h')
	if(line.find('naf2.h') != -1):
		line = line.replace('naf2.h', 'naf2CPU.h')

	if(line.find('__host__') != -1):
		line = line.replace('__host__', '')

	if(line.find('curandState') != -1):
		line = line.replace('curandState', 'int')

	print(line, file=file2, end='')

file1.close()
file2.close()


################################################
# Orbit2.cu 
################################################

filename = '../source/Orbit2.cu'
filename2 = 'Orbit2CPU.cu'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()


for line in Lines:
	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')

	if(line.find('__host__') != -1):
		line = line.replace('__host__', '')
	if(line.find('cudaSetDevice') != -1):
		continue
	if(line.find('cudaMalloc') != -1):
		continue
	if(line.find('cudaFree') != -1):
		continue
	if(line.find('cudaMemcpy') != -1):
		continue
	if(line.find('cudaMemset') != -1):
		line = line.replace('cudaMemset', 'memset')
		line = line.replace('_d', '_h')

	if(line.find('curandState') != -1):
		line = line.replace('curandState', 'int')

	#Add loops
	if(line.find('void BufferInit_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void randomInit_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void removeM_kernel') != -1):
		addLoop = 1

	if(line.find('void remove_kernel') != -1):
		addLoop = 1

	if(line.find('void remove3M_kernel') != -1):
		loop_id = 'idy'
		loop_N = 'N'
		loop_id2 = 'st'
		loop_N2 = 'Nst'
		addLoop = 1

	if(line.find('void remove4M_kernel') != -1):
		loop_id = 'id'
		loop_N = 'Nencpairs'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0

	#Remove functions

	if(remove == 1):
		if(line.startswith('}',0,1)):
			remove = 0
			continue

	if(remove == 1):
		continue

	if(line.find('BufferInit_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('randomInit_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('remove_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('removeM_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('remove3M_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('remove4M_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')


	line = DtoH(line)

	print(line, file=file2, end='')

file1.close()
file2.close()


################################################
# output.cu 
################################################

filename = '../source/output.cu'
filename2 = 'outputCPU.cu'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:

	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')

	if(line.find('cudaMemcpy') != -1):
		continue 

	if(line.find('cudaDeviceSynchronize') != -1):
		continue 

	if(line.find('cudaMemset') != -1):
		line = line.replace('cudaMemset', 'memset')
		line = line.replace('_d', '_h')


	#Add loops
	if(line.find('void CoordinateToBuffer_kernel') != -1):
		loop_id = 'id'
		loop_N = 'NT + NsmallT'
		addLoop = 1


	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0

	if(line.find('CoordinateToBuffer_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	line = DtoH(line)

	print(line, file=file2, end='')

	# no MCMC for cpu version
	# replace cuda events

file1.close()
file2.close()


################################################
# BSSingle.h
################################################

filename = '../source/BSSingle.h'
filename2 = 'BSSingleCPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:


	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('BSSINGLE_H') != -1):
		line = line.replace('BSSINGLE_H', 'BSSINGLECPU_H')

	if(line.find('directAcc.h') != -1):
		line = line.replace('directAcc.h', 'directAccCPU.h')

	print(line, file=file2, end='')

file1.close()
file2.close()


################################################
# FG2.h 
################################################

filename = '../source/FG2.h'
filename2 = 'FG2CPU.h'

file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

remove = 0

for line in Lines:

	#Remove functions
	if(line.find('void HCfg_kernel') != -1):
		remove = 1
	if(line.find('void fgM_kernel') != -1):
		remove = 1
	if(line.find('void fgMSimple_kernel') != -1):
		remove = 1

	if(remove == 1):
		if(line.startswith('}',0,1)):
			remove = 0
			continue

	if(remove == 1):
		continue


	if(line.find('FG2_H') != -1):
		line = line.replace('FG2_H', 'FG2CPU_H')

	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')

	if(line.find('BSSingle.h') != -1):
		line = line.replace('BSSingle.h', 'BSSingleCPU.h')

	if(line.find('__constant__') != -1):
		line = line.replace('__constant__ ', '')

	if(line.find('__noinline__') != -1):
		line = line.replace('__noinline__', '')

	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('__syncthreads') != -1):
		continue 

	#Add loops
	if(line.find('void PoincareSection_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void fg_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		omp = 1
		addLoop = 1

	if(line.find('void fgS_kernel') != -1):
		loop_id = 'idd'
		loop_N = 'Nencpairs3_d[0]'
		omp = 1
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0


	line = DtoH(line)
	
	print(line, file=file2, end='')


file1.close()
file2.close()

################################################
# Kick3.h 
################################################

filename = '../source/Kick3.h'
filename2 = 'Kick3CPU.h'

file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

remove = 0

for line in Lines:

	#Remove functions
	if(line.find('void acc_d') != -1):
		remove = 1
	if(line.find('void acc_df') != -1):
		remove = 1
	if(line.find('void kick32BM_kernel') != -1):
		remove = 1
	if(line.find('void kick32BMSimple_kernel') != -1):
		remove = 1
	if(line.find('void kick32BMTTV_kernel') != -1):
		remove = 1
	if(line.find('void kick32BMTTVSimple_kernel') != -1):
		remove = 1
	if(line.find('void CollectGPUsAb_kernel') != -1):
		remove = 1
	if(line.find('void kick32ATTV_kernel') != -1):
		remove = 1
	if(line.find('void accG3') != -1):
		remove = 1
	if(line.find('void kick32c_kernel') != -1):
		remove = 1
	if(line.find('void kick32cf_kernel') != -1):
		remove = 1
	if(line.find('void kick16c_kernel') != -1):
		remove = 1
	if(line.find('void kick16cf_kernel') != -1):
		remove = 1
	if(line.find('void KickM2_kernel') != -1):
		remove = 1
	if(line.find('void KickM2Simple_kernel') != -1):
		remove = 1
	if(line.find('void KickM2TTV_kernel') != -1):
		remove = 1
	if(line.find('void KickM3_kernel') != -1):
		remove = 1

	if(remove == 1):
		if(line.startswith('}',0,1)):
			remove = 0
			continue

	if(remove == 1):
		continue


	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('volatile') != -1):
		line = line.replace('volatile ', '')

	if(line.find('__syncthreads') != -1):
		continue 

	if(line.find('template') != -1):
		continue 

	#Add loops

	if(line.find('void kick32C_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void Sortb_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void SortSb_kernel') != -1):
		loop_id = 'idd'
		loop_N = 'Nencpairs3_d[0]'
		addLoop = 1

	if(line.find('void kick32Ab_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		#omp = 1
		addLoop = 1

	if(line.find('void kickS_kernel') != -1):
		loop_id = 'idd'
		loop_N = 'Nencpairs3_d[0]'
		omp = 1
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0


	line = DtoH(line)
	
	print(line, file=file2, end='')


file1.close()
file2.close()

################################################
# Kick4.h 
################################################

filename = '../source/Kick4.h'
filename2 = 'Kick4CPU.h'

file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

remove = 0

for line in Lines:

	#Remove functions
	if(line.find('void acc_e') != -1):
		remove = 1
	if(line.find('void acc_ef') != -1):
		remove = 1
	if(line.find('void acc4C_kernel') != -1):
		remove = 1
	if(line.find('void acc4Cf_kernel') != -1):
		remove = 1
	if(line.find('void forceij') != -1):
		remove = 1
	if(line.find('void accc') != -1):
		remove = 1
	if(line.find('void ForceTri_kernel') != -1):
		remove = 1
	if(line.find('void ForceDiag_kernel') != -1):
		remove = 1
	if(line.find('void ForceSq_kernel') != -1):
		remove = 1
	if(line.find('void EncpairsZero_kernel') != -1):
		remove = 1
	if(line.find('void acclargeN_kernel') != -1):
		remove = 1
	if(line.find('void ForceDriver') != -1):
		remove = 1

	if(remove == 1):
		if(line.startswith('}',0,1)):
			remove = 0
			continue

	if(remove == 1):
		continue


	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('__syncthreads') != -1):
		continue 

	if(line.find('template') != -1):
		continue 

	if(line.find('volatile') != -1):
		line = line.replace('volatile ', '')

	#Add loops

	if(line.find('void EncpairsZeroC_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void compare_a_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1


	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0


	line = DtoH(line)
	
	print(line, file=file2, end='')


file1.close()
file2.close()

################################################
# Scan.h 
################################################

filename = '../source/Scan.h'
filename2 = 'ScanCPU.h'

file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

remove = 0

for line in Lines:

	#Remove functions
	if(line.find('void Scan32d1_kernel') != -1):
		remove = 1
	if(line.find('void Scan32d2_kernel') != -1):
		remove = 1
	if(line.find('void Scan32d3_kernel') != -1):
		remove = 1
	if(line.find('void Scan32a_kernel') != -1):
		remove = 1
	if(line.find('void Scan32c_kernel') != -1):
		remove = 1

	if(remove == 1):
		if(line.startswith('}',0,1)):
			remove = 0
			continue

	if(remove == 1):
		continue


	line = DtoH(line)
	
	print(line, file=file2, end='')


file1.close()
file2.close()



################################################
# Rcrit.h 
################################################

filename = '../source/Rcrit.h'
filename2 = 'RcritCPU.h'

file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

remove = 0

for line in Lines:
	
	#Remove functions
	if(line.find('RcritM_kernel') != -1):
		remove = 1

	if(remove == 1 and line == '}\n'):
		remove = 0
		continue

	if(remove == 1):
		continue

	#Add loops
	if(line.find('void Rcritb_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void Rcrit_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void RcritS_kernel') != -1):
		loop_id = 'idd'
		loop_N = 'Nencpairs3_d[0]'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0


		
	if(line.find('__host__') != -1):
		line = line.replace('__host__ ', '')

	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	
	if(line.find('__syncthreads') != -1):
		continue 
	

	if(line.find('__restrict__') != -1):
		line = line.replace('__restrict__', '')
	line = DtoH(line)
	
	print(line, file=file2, end='')


file1.close()
file2.close()




################################################
# directAcc.h
################################################

filename = '../source/directAcc.h'
filename2 = 'directAccCPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:

	if(line.find('DIRECTACC_H') != -1):
		line = line.replace('DIRECTACC_H', 'DIRECTACCCPU_H')

	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('curandState') != -1):
		line = line.replace('curandState', 'int')

	line = DtoH(line)
	print(line, file=file2, end='')

file1.close()
file2.close()


################################################
# convert.h
################################################

filename = '../source/convert.h'
filename2 = 'convertCPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:

	if(line.find('CONVERT_H') != -1):
		line = line.replace('CONVERT_H', 'CONVERTCPU_H')

	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	print(line, file=file2, end='')

file1.close()
file2.close()



################################################
# force.h 
################################################

filename = '../source/force.h'
filename2 = 'forceCPU.h'

file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

remove = 0

for line in Lines:
	if(line.find('Host2.h') != -1):
		line = line.replace('Host2.h', 'Host2CPU.h')
	
	#Remove functions
	if(line.find('void forced2_kernel') != -1):
		remove = 1

	if(line.find('void forceM_kernel') != -1):
		remove = 1

	if(line.find('void forceBM_kernel') != -1):
		remove = 1

	if(remove == 1 and line == '}\n'):
		remove = 0
		continue

	if(remove == 1):
		continue

	#Add loops
	if(line.find('void force_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(line.find('void setElements_kernel') != -1):
		loop_id = 'id'
		loop_N = 'nbodies'
		addLoop = 1

	if(line.find('void rotation_kernel') != -1):
		loop_id = 'id'
		loop_N = 'Nsmall + N'
		addLoop = 1

	if(line.find('void fragment_kernel') != -1):
		loop_id = 'id'
		loop_N = 'Nsmall + N'
		addLoop = 1

	if(line.find('void CallYarkovsky2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(line.find('void PoyntingRobertsonEffect_averaged_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(line.find('void PoyntingRobertsonEffect2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(line.find('void artificialMigration_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(line.find('void artificialMigration2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0


		
	if(line.find('__host__') != -1):
		line = line.replace('__host__ ', '')

	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('__constant__') != -1):
		line = line.replace('__constant__ ', '')

	
	if(line.find('__syncthreads') != -1):
		continue 
	if(line.find('cudaDeviceSynchronize') != -1):
		continue 
	if(line.find('template <') != -1):
		continue

	if(line.find('curandState') != -1):
		line = line.replace('curandState', 'int')

	if(line.find('curand_uniform(&random)') != -1):
		line = line.replace('curand_uniform(&random)', 'drand48()')


	if(line.find('fragment_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('rotation_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')
	

	line = DtoH(line)
	
	print(line, file=file2, end='')


file1.close()
file2.close()


################################################
# forceYarkovskyOld.h
################################################

filename = '../source/forceYarkovskyOld.h'
filename2 = 'forceYarkovskyOldCPU.h'

file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

remove = 0

for line in Lines:
	if(line.find('Host2.h') != -1):
		line = line.replace('Host2.h', 'Host2CPU.h')
	
	if(line.find('void CallYarkovsky_averaged_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0


		
	if(line.find('__host__') != -1):
		line = line.replace('__host__ ', '')

	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('__constant__') != -1):
		line = line.replace('__constant__ ', '')

	
	if(line.find('__syncthreads') != -1):
		continue 
	if(line.find('cudaDeviceSynchronize') != -1):
		continue 
	if(line.find('template <') != -1):
		continue

	if(line.find('curandState') != -1):
		line = line.replace('curandState', 'int')

	if(line.find('curand_uniform(&random)') != -1):
		line = line.replace('curand_uniform(&random)', 'drand48()')


	if(line.find('fragment_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('rotation_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')
	

	line = DtoH(line)
	
	print(line, file=file2, end='')


file1.close()
file2.close()


################################################
# createparticles.h
################################################

filename = '../source/createparticles.h'
filename2 = 'createparticlesCPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:


	if(line.find('__host__') != -1):
		line = line.replace('__host__ ', '')

	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')

	if(line.find('void create1_kernel') != -1):
		loop_id = 'id'
		loop_N = 'NN'
		addLoop = 1

	if(line.find('void create2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'NN'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0

	if(line.find('cudaDeviceSynchronize') != -1):
		continue 

	if(line.find('curandState') != -1):
		line = line.replace('curandState', 'int')

	if(line.find('curand_uniform(&random)') != -1):
		line = line.replace('curand_uniform(&random)', 'drand48()')

	if(line.find('create1_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('create2_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	line = DtoH(line)
	line = memorycopy(line)

	print(line, file=file2, end='')

file1.close()
file2.close()


################################################
# gas.cu
################################################

filename = '../source/gas.cu'
filename2 = 'gasCPU.cu'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:


	if(line.find('__host__') != -1):
		line = line.replace('__host__ ', '')
	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')
	if(line.find('__constant__') != -1):
		line = line.replace('__constant__ ', '')

	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')

	if(line.find('cudaMalloc') != -1):
		continue 

	if(line.find('cudaFree') != -1):
		continue 

	#Remove functions
	if(line.find('void gasEnergyd1_kernel') != -1):
		remove = 1

	if(line.find('void gasEnergyd2_kernel') != -1):
		remove = 1

	if(line.find('void gasEnergya_kernel') != -1):
		remove = 1

	if(line.find('void gasEnergyc_kernel') != -1):
		remove = 1

	if(line.find('void gasEnergy_kernel') != -1):
		remove = 1

	if(line.find('void Data::gasEnergyMCall') != -1):
		remove = 1

	if(line.find('void Data::GasAccCall_M') != -1):
		remove = 1


	if(remove == 1 and line == '}\n'):
		remove = 0
		continue

	if(remove == 1):
		continue
	
	if(line.find('void GasDisk_kernel') != -1):
		loop_id = 'ig'
		loop_N = 'G_Nr_g'
		loop_id2 = 'jg'
		loop_N2 = 'def_Gasnz_g'
		addLoop = 1

	if(line.find('void gasTable_kernel') != -1):
		loop_id = 'ip'
		loop_N = 'G_Nr_p'
		loop_id2 = 'jp'
		loop_N2 = 'def_Gasnz_p'
		addLoop = 1

	if(line.find('void GasAcc_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1

	if(line.find('void GasAcc2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N + Nstart'
		addLoop = 1


	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0

	if(line.find('cudaDeviceSynchronize') != -1):
		continue 
	if(line.find('dim3') != -1):
		continue 
	if(line.find('__syncthreads') != -1):
		continue 

	if(line.find('Gas_rg_d') != -1):
		line = line.replace('Gas_rg_d', 'Gas_rg_h')
	if(line.find('Gas_zg_d') != -1):
		line = line.replace('Gas_zg_d', 'Gas_zg_h')
	if(line.find('Gas_rho_d') != -1):
		line = line.replace('Gas_rho_d', 'Gas_rho_h')
	if(line.find('GasDisk_d') != -1):
		line = line.replace('GasDisk_d', 'GasDisk_h')
	if(line.find('GasAcc_d') != -1):
		line = line.replace('GasAcc_d', 'GasAcc_h')

	if(line.find('GasDisk_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('gasTable_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('GasAcc_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('GasAcc2_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')
	line = DtoH(line)
	line = memorycopy(line)

	print(line, file=file2, end='')

file1.close()
file2.close()


################################################
# Encounter3.h
################################################

filename = '../source/Encounter3.h'
filename2 = 'Encounter3CPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:


	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')

	if(line.find('Orbit2.h') != -1):
		line = line.replace('Orbit2.h', 'Orbit2CPU.h')


	#Remove functions
	if(line.find('void setNencpairs2_kernel') != -1):
		remove = 1
	if(line.find('void encounterM_kernel') != -1):
		remove = 1
	if(line.find('void encounterM3_kernel') != -1):
		remove = 1
	if(line.find('void groupM2_kernel') != -1):
		remove = 1
	if(line.find('void groupM3_2_kernel') != -1):
		remove = 1
	if(line.find('void Data::groupCall') != -1):
		remove = 1
	if(line.find('void group_kernel') != -1):
		remove = 1
	if(line.find('void groupM1_kernel') != -1):
		remove = 1
	if(line.find('void groupM3_kernel') != -1):
		remove = 1


	if(remove == 1 and line == '}\n'):
		remove = 0
		continue

	if(remove == 1):
		continue
	
	if(line.find('void setNencpairs_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void encounter_kernel') != -1):
		loop_id = 'id'
		loop_N = 'Nencpairs'
		omp = 1
		addLoop = 1

	if(line.find('void encounter_small_kernel') != -1):
		loop_id = 'id'
		loop_N = 'Nencpairs2'
		addLoop = 1

	if(line.find('void setEnc3_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void groupS2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'Ne'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0

	if(line.find('__syncthreads') != -1):
		continue 
	if(line.find('template') != -1):
		continue 

	line = DtoH(line)

	print(line, file=file2, end='')

file1.close()
file2.close()

################################################
# bvh.h
################################################

filename = '../source/bvh.h'
filename2 = 'bvhCPU.h'


file1 = open(filename, 'r')
file2 = open(filename2, 'w')

Lines = file1.readlines()

for line in Lines:


	if(line.find('__device__') != -1):
		line = line.replace('__device__ ', '')


	#Remove functions
	if(line.find('void sort_kernel') != -1):
		remove = 1

	if(line.find('void sortmerge_kernel') != -1):
		remove = 1

	if(line.find('void sortscatter_kernel') != -1):
		remove = 1

	if(line.find('void sortscatter2_kernel') != -1):
		remove = 1

	if(remove == 1 and line == '}\n'):
		remove = 0
		continue

	if(remove == 1):
		continue
	
	if(line.find('void collisioncheck_kernel') != -1):
		loop_id = 'idx'
		loop_N = 'N'
		loop_id2 = 'idy'
		loop_N2 = 'N'
		addLoop = 1

	if(line.find('void sortscatter_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void sortscatter2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void sortCheck_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void sortCheck2_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void setLeafNode_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void setInternalNode_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void checkNodes_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N - 1'
		addLoop = 1

	if(line.find('void buildBVH_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(line.find('void traverseBVH_kernel') != -1):
		loop_id = 'id'
		loop_N = 'N'
		addLoop = 1

	if(addLoop == 1):
		line, i_id, i_id2 = kernelToLoop(line, i_id, loop_id, loop_N, omp, i_id2, loop_id2, loop_N2)
		if(line.startswith('}',0,1)):
			addLoop = 0
			loop_id = ''
			loop_id2 = ''
			loop_N = ''
			loop_N2 = ''
			i_id = ''
			i_id22 = ''
			omp = 0

	if(line.find('__syncthreads') != -1):
		continue 

	if(line.find('__threadfence') != -1):
		continue 

	if(line.find('template') != -1):
		continue 

	if(line.find('collisioncheck_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('setLeafNode_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('setInternalNode_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('buildBVH_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('traverseBVH_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('sortCheck2_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')

	if(line.find('checkNodes_kernel') != -1):
		line = line.replace('_kernel', '_cpu')
		line = line.replace('<<<', '/*')
		line = line.replace('>>>', '*/')


	line = DtoH(line)

	print(line, file=file2, end='')

file1.close()
file2.close()



