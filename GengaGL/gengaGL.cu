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
static GLdouble anglex = 0.0;	//angle to rotate system around z axis
static GLdouble angley = 0.0;	//angle to rotate z axis
static GLdouble omegax = 0.0;	//speed to rotate around z axis to fix planet positions
static GLdouble movex = 0.0;	//shift along x axis
static GLdouble movey = 0.0;	//shift along y axis
static GLdouble zoom = 1.0;
static GLdouble izoom = 1.0;
static GLdouble pointsize = 2.0;
GLdouble yold = 0.0;
GLdouble xold = 0.0;
GLint mouseMove = 0;
GLint translate = 0;
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
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	double time;
	time = D.time_h[0] / 365.25;

	glEnable(GL_LINE_SMOOTH);

	//print time and length scale
	glPushMatrix();
	glColor3d(0.0, 100.0, 0.0);
	glScaled(0.002, 0.002, 0.002);
	glTranslated(1400.0, -2800, 0.0);
	sprintf(mtext, "%6.2g AU", izoom);
	for(int i = 0; i < 10; ++i){
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, mtext[i]);
	}
	glColor3d(1.0, 1.0, 1.0);
	glTranslated(-5100, 0.0, 0.0);
	sprintf(mtext, "%10g yr", time);
	for(int i = 0; i < 14; ++i){
		glutStrokeCharacter(GLUT_STROKE_MONO_ROMAN, mtext[i]);
	}
	glPopMatrix();

	glTranslated(movex, movey, 0.0);
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


	if(stop == 0){
		for(int i = 0; i < 1; ++i){
			D.timeStep = ts;
			int er = D.timeStepLoop(interrupted, 0);
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
	if(translate == 0){
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
	else{
		if(mouseMove == 1){
			double dy = yold - y;
			movey += dy * 0.1;
			yold = y;
			double dx = xold - x;
			movex -= dx * 0.1;
			xold = x;
		}
	}
}

void keyPressed (unsigned char key, int x, int y){
	if (key == 'r') pointsize += 1.0;
	if (key == 't') pointsize -= 1.0;
	if(pointsize < 1) pointsize = 1.0;
	if (key == 'p'){
		if(stop == 0) stop = 1;
		else stop = 0;
	}
	if (key == ' '){
		if(stop == 0) stop = 1;
		else stop = 0;
	}
	if (key == 'a') translate = 1;
	if (key == 'o'){
		movex = 0.0;
		movey = 0.0;
		anglex = 0.0;
		angley = 0.0;
		omegax = 0.0;
		zoom = 1.0;
		izoom = 1.0;
		pointsize = 2.0;
	}
}
void keyUp (unsigned char key, int x, int y){
	if (key == 'a'){
		translate = 0;
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
	glutKeyboardUpFunc(keyUp);

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
	error = cudaSetDevice(H.P.dev[0]);
//	fprintf(D.masterfile, "Set device error = %d = %s\n",error, cudaGetErrorString(error));

	//Allocate memory for parameters on the device:
	H.Calloc();
	H.Info();

	//Determine the start points of the individual simulations
	H.Tsizes();

//	Data D = H;
	D = H;

	er = D.beforeTimeStepLoop1();
	if(er == 0) return 0;

	er = D.beforeTimeStepLoop(0);	
	if(er == 0) return 0;

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


	
	// *************************************************************3
	// **************************************************************
	// time step loop

	glutMainLoop();

	// **************************************************************
	// **************************************************************

	er = D.Remaining();
	if(er == 0) return 0;

	return 0; 
}
