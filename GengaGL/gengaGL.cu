/**************************************
* This is a modification of the genga.cu file. It includes
* the openGL interop 
*
* use the MakefileGL to compile it
*
* Authors: Simon Grimm
* January 2015
*
****************************************/

#include "../source/define.h"
#include "../source/Host2.h"
#include "../source/Orbit2.h"


//openGL
#define GL_GLEXT_PROTOTYPES
#include <GL/freeglut.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include "signal.h"
//#include "glext.h"
GLuint positionsVBO;
GLuint colorsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;
struct cudaGraphicsResource* colorsVBO_CUDA;
static long long ts = 1;
static GLdouble anglex = 0.0; //angle to rotate system around z axis
static GLdouble angley = 0.0; //angle to rotate z axis
static GLdouble omegax = 0.0; // speed to rotate around z axis to fix planet positions
static GLdouble zoom = 1.0;
static GLdouble izoom = 1.0;
static GLdouble pointsize = 1.5;
GLdouble yold = 0.0;
GLdouble xold = 0.0;
GLint mouseMove = 0;
GLint nold = 0;
static GLint stop = 0;

char mtext[80];

Data D(0);
double *MLimits_d;
/*
#include "BS.h"
//#include "BSA.h"
*/
volatile sig_atomic_t interrupted = 0;

void catch_signal(int sig){
	interrupted = 1;
}


__global__ void GLPositions(double4 *x4_d, double4 *positions, double4 *colors, int N, int Nsmall, double *MLimits_d, double MinMass){

	int id = blockIdx.x * blockDim.x + threadIdx.x;

	double4 x;
	if(id < N + Nsmall){
		x = x4_d[id];
	}
	else{
		x.x = 0.0;
		x.y = 0.0;
		x.z = 0.0;
		x.w = 1.0;
	}

	double mscale = MLimits_d[0] / x.w;

//	x.w /= MLimits_d[0];

	double4 color;
	if(x.w > MinMass){
		color.x = 1.0;
		color.y = mscale;
		color.z = 0.0;
		color.w = 1.0;
	}
	else{
		color.x = 1.0;
		color.y = 1.0;
		color.z = 1.0;
		color.w = 1.0;
	}
	if(x.w < 0.0){
		color.x = 0.0;
		color.y = 0.0;
		color.z = 0.0;
		color.w = 0.0;
	}

	if(id < N + Nsmall){
		positions[id] = x;
		colors[id] = color;
	}


}

__global__ void GLMLimits(double4 *x4_d, int N, double *MLimits_d){

	int idy = threadIdx.x;

	__shared__ double mmin_s[512];
	__shared__ double mmax_s[512];

	mmin_s[idy] = 1.0e30;
	mmax_s[idy] = 0.0;

	__syncthreads();

	for(int k = 0; k < N; k += 512){
		if(idy + k < N){
			double m = x4_d[idy + k].w;
			mmin_s[idy] = fmin(mmin_s[idy], m);
			mmax_s[idy] = fmax(mmax_s[idy], m);
		}
	}
	__syncthreads();

	if(N >= 512){
		if(idy < 256){
			mmin_s[idy] = fmin(mmin_s[idy], mmin_s[idy + 256]);
			mmax_s[idy] = fmax(mmax_s[idy], mmax_s[idy + 256]);
		}
	}
	__syncthreads();
	if(N >= 256){
		if(idy < 128){
			mmin_s[idy] = fmin(mmin_s[idy], mmin_s[idy + 128]);
			mmax_s[idy] = fmax(mmax_s[idy], mmax_s[idy + 128]);
		}
	}
	__syncthreads();
	if(N >= 128){
		if(idy < 64){
			mmin_s[idy] = fmin(mmin_s[idy], mmin_s[idy + 64]);
			mmax_s[idy] = fmax(mmax_s[idy], mmax_s[idy + 64]);
		}
	}
	__syncthreads();

	if(idy < 32){
		volatile double *mmin = mmin_s;
		volatile double *mmax = mmax_s;
		mmin[idy] = fmin(mmin[idy], mmin[idy + 32]);
		mmax[idy] = fmax(mmax[idy], mmax[idy + 32]);
		mmin[idy] = fmin(mmin[idy], mmin[idy + 16]);
		mmax[idy] = fmax(mmax[idy], mmax[idy + 16]);
		mmin[idy] = fmin(mmin[idy], mmin[idy + 8]);
		mmax[idy] = fmax(mmax[idy], mmax[idy + 8]);
		mmin[idy] = fmin(mmin[idy], mmin[idy + 4]);
		mmax[idy] = fmax(mmax[idy], mmax[idy + 4]);
		mmin[idy] = fmin(mmin[idy], mmin[idy + 2]);
		mmax[idy] = fmax(mmax[idy], mmax[idy + 2]);
		mmin[idy] = fmin(mmin[idy], mmin[idy + 1]);
		mmax[idy] = fmax(mmax[idy], mmax[idy + 1]);
	}
	if(idy == 0){
		MLimits_d[0] = mmin_s[0];
		MLimits_d[1] = mmax_s[0];
	}

}


void display(){

	int modx = int(anglex) % 360;
	anglex -= modx * 360.0;

	anglex -= omegax;

        glLoadIdentity();
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0, 1, 0);
        glRotated(angley, 1.0, 0.0, 0.0);
        glRotated(anglex, 0.0, 0.0, 1.0);
	glScaled(zoom, zoom, zoom);

	double4 *positions;
	double4 *colors;
	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);

	cudaGraphicsMapResources(1, &colorsVBO_CUDA, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&colors, &num_bytes, colorsVBO_CUDA);


	double time;
	time = D.time_h[0] / 365.25;
	if(stop == 0){
		for(int i = 0; i < 1; ++i){
			D.timeStep = ts;
			int er = D.timeStepLoop(interrupted);
			time = D.time_h[0] / 365.25;
			++ts;
		}
	}
	if(D.N_h[0] + D.Nsmall_h[0] != nold){
		GLMLimits <<< 1, 512 >>>(D.x4_d, D.N_h[0], MLimits_d);
	}
	nold = D.N_h[0] + D.Nsmall_h[0];

	int nb = (D.N_h[0] + D.Nsmall_h[0] + 255) / 256; 

	//Fill the position array from the CUDA arrays
	GLPositions <<< nb, 256 >>> (D.x4_d, positions, colors, D.N_h[0], D.Nsmall_h[0], MLimits_d, D.P.MinMass);
	cudaDeviceSynchronize();
	// Unmap buffer object
	cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
	cudaGraphicsUnmapResources(1, &colorsVBO_CUDA, 0);

	// Render from buffer object
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_DOUBLE, 32, 0);


//glEnableClientState(GL_POINT_SIZE_ARRAY_OES);
//glPointSizePointer(1, GL_DOUBLE, 32, ((void *) NULL + (24)));

	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(4, GL_DOUBLE, 32, 0);


glPointSize(pointsize);

	glDrawArrays(GL_POINTS, 0, D.N_h[0] + D.Nsmall_h[0]);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
//glDisableClientState(GL_POINT_SIZE_ARRAY_OES);

	anglex += omegax;
glPointSize(1.0);

	//Draw Axis in scale
	glColor3d(0, 100, 0);
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(izoom, 0.0, 0.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(0.0, izoom, 0.0);
	glEnd();
	glBegin(GL_LINES);
	glVertex3d(0.0, 0.0, 0.0);
	glVertex3d(0.0, 0.0, izoom);
	glEnd();

	glEnable(GL_LINE_SMOOTH);

	//Print time
	glColor3d(1.0, 1.0, 1.0);
	glRotated(-anglex, 0.0, 0.0, 1.0);
	glRotated(-angley, 1.0, 0.0, 0.0);
	glScaled(izoom, izoom, izoom);
	glScaled(0.002, 0.002, 0.002);
	glTranslated(-2800.0, -2800.0, 0.0);

	sprintf(mtext, "%10g yr", time);
	for(int i = 0; i < 14; ++i){
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, mtext[i]);
	}

	glTranslated(2800.0, 0.0, 0.0);
	sprintf(mtext, "%6.2g AU", izoom);
	for(int i = 0; i < 10; ++i){
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, mtext[i]);
	}
	// Swap buffers

	anglex -= omegax;
	glutSwapBuffers();
	glutPostRedisplay();

}
//When window is resized
void reshape(int w, int h){

	glViewport (0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION); //Matrix is used for projection
	glLoadIdentity();

	gluPerspective(60.0, 1.0, 0.0, 1.0);
	glMatrixMode(GL_MODELVIEW); //Matrix is used for Data modelling
}


void mouse(int button, int state, int x, int y){
	switch (button) {
		case GLUT_LEFT_BUTTON:
			if (state == GLUT_DOWN){
				xold = x;
				yold = y;
				mouseMove = 1;
			}
			else mouseMove = 0;
			break;
		case GLUT_RIGHT_BUTTON:
			if (state == GLUT_DOWN){
				yold = y;
				mouseMove = 2;
			}
			else mouseMove = 0;
			break;
		case GLUT_MIDDLE_BUTTON:
			if (state == GLUT_DOWN){
				if(stop == 0) stop = 1;
				else stop = 0;
			}
			else mouseMove = 0;
			break;
		case 3:
			zoom /= 0.95;
			izoom *= 0.95;
			glutPostRedisplay();
			break;
		case 4:
			zoom *= 0.95;
			izoom /= 0.95;
			glutPostRedisplay();
			break;
		default:
			break;
	}
}
void mouse_move (int x, int y){
	if(mouseMove == 1){
		double dy = yold - y;
		angley -= dy * 0.2;
  		yold = y;
		double dx = xold - x;
		anglex -= dx * 0.2;
  		xold = x;
	}
	if(mouseMove == 2){
		double dy = yold - y;
		omegax -= dy * 0.001;
  		yold = y;
	}
}

void keyPressed (unsigned char key, int x, int y){
	if (key == 'r') pointsize *= 0.95;
	if (key == 't') pointsize /= 0.95;
	if (key == 'p'){
		if(stop == 0) stop = 1;
		else stop = 0;
	}
	if (key == ' '){
		if(stop == 0) stop = 1;
		else stop = 0;
	}
}


int main(int argc, char*argv[]){

	glutInit(&argc, argv);
	glutInitDisplayMode (GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize (800, 800);
	glutInitWindowPosition (100, 100);
	glutCreateWindow ("GENGA");

	glutDisplayFunc(display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mouse);
	glutMotionFunc(mouse_move);
	glutKeyboardFunc(keyPressed);

	cudaError_t error;

	long long Restart = 0LL;
	int RRestart = 0;
	//Check if simulation is restarted
	for(int i = 1; i < argc; i += 2){
		if(strcmp(argv[i], "-R") == 0){
			Restart = atol(argv[i + 1]);
			RRestart = 1;
		}
	}

	Data H(Restart);

	if(H.Lock == 1){
		printf("lock.dat file already exists for the current start time. Delete or modify the file to continue\n");
		fprintf(H.masterfile, "lock.dat file already exists for the current start time. Delete or modify the file to continue\n");
		return 0;

	}

	if(RRestart == 0){
       		printf("Start GENGA\n");
        	fprintf(H.masterfile,"Start GENGA\n");
	}
	if(RRestart == 1){
       		printf("Restart GENGA\n");
		fprintf(H.masterfile,"\n \n **************************************** \n \n");
        	fprintf(H.masterfile,"Restart GENGA\n");
	}

#if SERIAL_GROUPING > 0
	printf("Using serial grouping!\n");
	fprintf(H.masterfile, "Using serial grouping!\n");
#endif
	//determine the number of simulations
	int Nst = H.NSimulations(argc, argv);
	if(Nst == 0) return 0;

	//Check Device Informations
	int DevError = H.DeviceInfo();
	if(DevError == 0) return 0;

	//Allocate memory for parameters on the host:
	H.Halloc();

	// Read parameters from param file //
	printf("Read parameters\n");
	int er = H.Param(argc, argv);
	if(er == 0) return 0;
	printf("Parameters OK\n");

	// Determine the size of the simulations
	printf("Read Size\n");
	er = H.size();
	if(er == 0) return 0;
	
	printf("Size OK\n");
	cudaDeviceSynchronize();
//	cudaDeviceReset();
	error = cudaSetDevice(H.P.dev);
//	fprintf(D.masterfile, "Set device error = %d = %s\n",error, cudaGetErrorString(error));

	//Allocate memory for parameters on the device:
        H.Calloc();
	H.Info();

	//Determine the start points of the individual simulations
	H.Tsizes();

//	Data D = H;
	D = H;

	//Allocate orbit data on Host and Device
	D.AllocateOrbitt();

	 //allocate mapped memory//
	er = D.CMallocateOrbit();
	if(er == 0) return 0;

        //Allocate Grideae 
	D.constantCopy2();
	if(H.P.UseaeGrid == 1){
      		er = D.GridaeAlloc();
       		if(er == 0) return 0;
	}
	if(D.P.Usegas == 1){
		D.GasAlloc();
	}

	//Table for fastfg//
	er = D.FGAlloc();
	if(er == 0) return 0;

	//initialize memory//
	er = D.init();
	printf("\nInitialize Memory\n");	

	cudaDeviceSynchronize();
	//read initial conditions//
	printf("\nRead Initial Conditions\n");
	er = D.ic();
	if(er == 0) return 0;
	printf("Initial Conditions OK\n");

#if USE_NAF == 1
	er = D.naf.alloc1(D.NT, D.N_h[0], D.Nsmall_h[0], D.Nst, D.P.tRestart, D.idt_h, D.ict_h, D.P.NAFn0, D.P.NAFnfreqs);
	if(er == 0) return 0;

	er = D.naf.alloc2(D.NT, D.N_h[0], D.Nsmall_h[0], D.Nst, D.GSF, D.P.NAFformat, D.P.tRestart, D.index_h);
	if(er == 0) return 0;
#endif

	//remove ghost particles and reorder arrays//
	int NminFlag = D.remove();

	//remove stopped simulations//
	if(NminFlag == 1){
		D.stopSimulations();
		NminFlag = 0;
	}

	cudaDeviceSynchronize();
	printf("Compute initial Energy\n");

	er = D.firstEnergy();
	if(er == 0) return 0;

	cudaDeviceSynchronize();

	printf("Write initial Energy\n");

	//write first output
	er = D.firstoutput(0);
 	if(D.P.IrregularOutputs == 1){
		er = D.firstoutput(1);
	}
	if(er == 0) return 0;
	printf("Energy OK\n");

	//read aeGrid at restart time step 
	if(H.P.UseaeGrid == 1){
		D.readGridae();	
	}

	//Set Gas Disc and Gas Table
	if(D.P.Usegas == 1){
		printf("Set Gas Table\n");
		er = D.setGasDisk();
		if(er == 0) return 0;
		printf("Gas Table OK\n");
	}

	// Set Order and Coefficients of the symplectic integrator //
	D.SymplecticP(0);

	cudaDeviceSynchronize();
	cudaMemset(D.Energy_d, 0, D.NEnergyT*sizeof(double));
	if(Nst == 1) printf("Start integration with %d simulation\n", Nst);
	else printf("Start integration with %d simulations\n", Nst);
        error = cudaGetLastError();
	if(error != 0){
		fprintf(D.masterfile, "Start error = %d = %s\n",error, cudaGetErrorString(error));
        	printf("Start error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}

	fflush(D.masterfile);
#if USE_NAF == 1
	//compute the x and y arrays for the naf algorithm
	int NAFstep = 0;
	D.naf.getnafvarsCall(D.x4_d, D.v4_d, D.index_d, D.NBS_d, D.vcom_d, D.test_d, D.P.NAFvars, D.naf.x_d, D.naf.y_d, D.Msun_d, D.Msun_h[0].x, D.NT, D.Nst, D.naf.n, NAFstep, D.NB[0], D.N_h[0], D.Nsmall_h[0], D.P.UseTestParticles);
	++NAFstep;
#endif
	if(D.Nst > 1){
		D.firstKick_M(0);
	}
	else{
		if(D.P.UseTestParticles == 1) D.firstKick_small();
		else{
			switch( D.NB[0] ) {
				case 16: D.firstKick_16();
				break;
				case 32: D.firstKick_32();
				break;
				case 64: D.firstKick_64();
				break;
				case 128: D.firstKick_128();
				break;
				case 256: D.firstKick_256();
				break;
				case 512: D.firstKick_512();
				break;
				case 1024: D.firstKick_1024();
				break;
				case 2048: D.firstKick_2048();
				break;
			}
			if(D.NB[0] > 2048) D.firstKick_largeN();
		}
	}
	cudaDeviceSynchronize();
	//Print first informations about close encounter pairs
	D.firstInfo();
	D.setStartTime();
#if poincareFlag == 1
	sprintf(D.poincarefilename, "%sPoincare%s_%.12ld.dat", D.GSF[0].path, D.GSF[0].X, 0);
	D.poincarefile = fopen(D.poincarefilename, "w");
#endif

	D.irrTimeStep = 0;
	if(D.P.IrregularOutputs == 1){
		er = D.readIrregularOutputs();
		if(er == 0){
			return 0;
		}
	}
	if(D.P.setElements == 1){
		er = D.readSetElements();
		if(er == 0){
			return 0;
		}
	}
	
	D.MultiSim = 0;
	if(D.Nst > 1) D.MultiSim = 1;

	//Start time loop here
	D.timeStep = D.P.tRestart + 1;
	ts = D.timeStep;

	cudaDeviceSynchronize();
      //  cudaDeviceReset();
      //  error = cudaGLSetGLDevice(0);
	printf("GLDevice error = %d = %s\n",error, cudaGetErrorString(error));
	cudaDeviceSynchronize();

	// Create buffer object and register it with CUDA
	glGenBuffers(1, &positionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	unsigned int size = (D.N_h[0] + D.Nsmall_h[0] + def_Nfragments) *  sizeof(double4);

	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	error = cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);

	if(error != 0){
		fprintf(D.masterfile, "GL position error = %d = %s\n",error, cudaGetErrorString(error));
        	printf("GL position error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}

	glGenBuffers(1, &colorsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, colorsVBO);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	error = cudaGraphicsGLRegisterBuffer(&colorsVBO_CUDA, colorsVBO, cudaGraphicsMapFlagsWriteDiscard);

	if(error != 0){
		fprintf(D.masterfile, "GL color error = %d = %s\n",error, cudaGetErrorString(error));
        	printf("GL color error = %d = %s\n",error, cudaGetErrorString(error));
		return 0;
	}
	cudaMalloc((void **) &MLimits_d, 2 * sizeof(double));

	glutMainLoop();


#if poincareFlag == 1
	fclose(D.poincarefile);
#endif

	//print last informations
	D.printLastTime();
	D.LastInfo();

	//free all the memory on the Host and on the Device
	er = D.freeOrbit();
	if(er == 0) return 0;

#if useGridae
	free(D.Gridaecount_h);
	cudaFree(D.Gridaecount_d);
#endif

#if useGas > 0
	er = D.freeGas();
	if(er == 0) return 0;
#endif
	er = H.freeHost();
	if(er == 0) return 0;

        printf("GENGA terminated successfully\n");
	fprintf(H.masterfile, "GENGA terminated successfully\n");

	return 0; 
}
