//first order Post Newtonian Correction
//See Fabrycky or Kidder 1995

__global__ void force(double4 *x4_d, double4 *v4_d, double3 love_d, double4 Msun4, double4 Spinsun, double dt){

	int idy = threadIdx.x;
	int id = blockIdx.x * blockIdx.x + idy;

	double4 x4 = x4_d[id];
	double4 v4 = v4_d[id];
	double3 love = love_d[id];
	
	double3 a3;
	a3.x = 0.0;
	a3.y = 0.0;
	a3.z = 0.0;	


	// GR
	double c = 10065.3201686;//c in AU / day * 0.0172020989
	double csq = c * c;
	double Msun = Msun4.x;

	double rsq = (x4.x * x4.x + x4.y * x4.y + x4.z * x4.z);
	double ir = 1.0 / sqrt(rsq);
	double vsq = (v4.x * v4.x + v4.y * v4.y + v4.z * v4.z);
	double rd = (x4.x * v4.x + x4.y * v4.y + x4.z * v4.z) * ir; 

	double A = ksq * (Msun + x4.w) * ir;
	double B = A * ir / csq;

	double eta = Msun * x4.w / ((Msun + x4.w) * (Msun + x4.w));

	double C = 2.0 * (2.0 - eta) * rd;
	double D = (1.0 + 3.0 * eta) * vsq - 1.5 * eta * rd * rd - 2.0 * (2.0 + eta) * A;

	a3.x += B * (C * v4.x - D * x4.x * ir); 	
	a3.y += B * (C * v4.y - D * x4.y * ir);
	a3.z += B * (C * v4.z - D * x4.z * ir);


	//Tidal force
	double R2 = v4.w * v4.w;
	double R5 = R2 * R2 * v4.w;
	double ir3 = ir * ir * ir;
	double ir7 = ir3 * ir3 * ir;

	double E = -3.0 * love.x * ksq * Msun * Msun * R5 / x4.w * ir7;

	a3.x += E * x4.x * ir;
	a3.y += E * x4.y * ir;	
	a3.z += E * x4.z * ir;

	//Rotational Force
	double Rsun2 = Msun4.y * Msun4.y;
	double Rsun5 = Rsun2 * Rsun2 * Msun4.y;
	double lovesun = Msun4.z;
	double ir2 = ir * ir;
	double ir4 = ir2 * ir2;

	double3 omega
	//compute rotation vector from spin vector
	double iI = 5.0 / (2.0 * Msun * Rsun2); // inverse Moment of inertia of a solid sphere in 1/ (Solar Masses AU^2)
	double3 omega3;
	omega3.x = Spinsun.x * iI;
	omega3.y = Spinsun.y * iI;
	omega3.z = Spinsun.z * iI;

	double omega2 = omega3.x * omega3.x + omega3.y * omega3.y + omega3.z * omega3.z; 	//angular velocity in 1 / day * 0.017

	double F = -0.5 * lovesun * omega2 * Rsun5 * ir4;

	a3.x += F * x4.x * ir;
	a3.y += F * x4.y * ir;	
	a3.z += F * x4.z * ir;

        //Kick
	v4.x += a3.x * dt;
	v4.y += a3.y * dt;
	v4.z += a3.z * dt;

	v4_d[id] = v4;

}
