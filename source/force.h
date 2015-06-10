// *******************************************************
// This is a template function for additional forces
// The velocities in this kernel are already converted to heliocentric coordinates
//
//June 2015
//Authors: Simon Grimm
// **********************************************************
__global__ void force(double4 *x4_d, double4 *v4_d, int *index_d, double *Msun_d, double *dt_d, double Ct, double *time_d, int N, int Nst, int UseForce){
/*
	int idy = threadIdx.x;
	int id = blockIdx.x * blockIdx.x + idy;

	if(id < N){
		
		int st = 0;

		if(Nst > 1 && id < N) st = index_d[id] / 100;	//st is the sub simulation index

		double4 x4 = x4_d[id];
		double4 v4 = v4_d[id];
		int index = index_d[id];
		double Msun = Msun_d[st];			//This is the mass of the central star
		double dt = dt_d[st] * Ct;			//This is the time step to do
		double time = time_d[st] / 365.25;		//This is the time in years

		double3 a3;
		a3.x = 0.0; 	
		a3.y = 0.0;
		a3.z = 0.0;
		
		if(UseForce == 1){
			//Insert here the force 	
			
		}

		//apply the Kick
		v4.x += a3.x * dt;
		v4.y += a3.y * dt;
		v4.z += a3.z * dt;

		v4_d[id] = v4;
	}
*/
}

