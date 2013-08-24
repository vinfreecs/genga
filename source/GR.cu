//first order Post Newtonian Correction
//See Fabrycky or Kidder 1995

__global__ void force(double4 *x4_d, double4 *v4_d, double Msun, double csq, double dt){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockIdx.x + idy;

	double4 x4 = x4_d[id];
	double4 v4 = v4_d[id];

	double rsq = (x4.x * x4.x + x4.y * x4.y + x4.z * x4.z);
	double ir = 1.0 / sqrt(rsq);
	double vsq = (v4.x * v4.x + v4.y * v4.y + v4.z * v4.z);
	double rd = (x4.x * v4.x + x4.y * v4.y + x4.z * v4.z) * ir; 

	double A = ksq * (Msun + x4.w) * ir;
	double B = -A * ir / csq / x4.w;

	double eta = Msun * x4.w / ((Msun + x4.w) * (Msun + x4.w));

	double C = -2.0 * (2.0 - eta) * rd;
	double D = (1.0 + 3.0 * eta) * vsq - 1.5 * eta * rd * rd - 2.0 * (2.0 + eta) * A;


	double3 a3;

	a3.x = B * (C * v4.x + D); 	
	a3.y = B * (C * v4.y + D);
	a3.z = B * (C * v4.z + D);

        //Kick
	v4.x += a3.x * dt;
	v4.y += a3.y * dt;
	v4.z += a3.z * dt;

	v4_d[id] = v4;

}
