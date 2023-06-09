#include "directAccCPU.h"
#include "Encounter3CPU.h"

template< int NN>
void BSBStep_cpu(int *random_h, double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, double *rcrit_h, double *rcritv_h, int2 *Encpairs2_h, const double dt, const double Msun, double *U_h, const int st, int *index_h, int *BSstop_h, int *Ncoll_m, double *Coll_h, const double time, double4 *spin_h, double3 *love_h, int *createFlag_h, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *enccount_h, unsigned long long *aecountT_h, unsigned long long *enccountT_h, const int NT, const int NconstT, int *NWriteEnc_m, double *writeEnc_h, const int UseGR, const double MinMass, const int UseTestParticles, const int SLevels, int noColl, int idx){


	int random = 0;
//printf("BSB start %d %d\n", NN, idx);
	if((noColl == 1 || noColl == -1) && BSstop_h[0] == 3){
//if(idx == 0)      printf("Stop BSB b\n");
		return;
	}
//printf("BSB %d %g %g %d\n", idx, StopMinMass_c[0], CollisionPrecision_c[0], noColl);

	double dt1 = dt; 
	double dt2, dt22;
	double t = 0.0;

	double dtgr[NN];

	double4 x4_s[NN];
	double4 v4_s[NN];
	double rcritv_s[NN * def_SLevelsMax];
	double3 a0_s[NN];
	double3 a_s[NN];
	double4 xp_s[NN];
	double4 vp_s[NN];
	double4 xt_s[NN];
	double4 vt_s[NN];
	double3 dx_s[NN][8];
	double3 dv_s[NN][8];

	int Ncol_s[1];
	int2 Colpairs_s[def_MaxColl];
	double Coltime_s[def_MaxColl];
	int sgnt;

	double3 scalex_s[NN];
	double3 scalev_s[NN];

	double error = 0.0;
	double test;
	int si = Encpairs2_h[ (st+2) * NT + idx].y;	//group index 
	int N2 = Encpairs2_h[si].y; //Number of bodies in current BS simulation
	int start = Encpairs2_h[NT + si].y;

//printf("BS %d %d %d %d %d %d\n", idx, st, si, N2, NT, start);

	if(dt < 0.0){
		sgnt = -1;
	}
	else sgnt = 1;

	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
//printf("BS2 %d %d %d %d %d %d %d\n", idx, st, si, idi, index_h[idi], N2, NN);

		x4_s[i] = xold_h[idi];
		v4_s[i] = vold_h[idi];
		for(int l = 0; l < SLevels; ++l){
			rcritv_s[i + l * NN] = rcritv_h[idi + l * NconstT];
		}
//printf("BSold %d %.40g %.40g %.40g %.40g %.40g %.40g\n", idi, xold_h[idi].x, xold_h[idi].y, xold_h[idi].z, vold_h[idi].x, vold_h[idi].y, vold_h[idi].z);
		if(UseGR == 1){// GR time rescale (Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			double mu = def_ksq * Msun;
			double rsq = x4_s[i].x * x4_s[i].x + x4_s[i].y * x4_s[i].y + x4_s[i].z * x4_s[i].z;
			double vsq = v4_s[i].x * v4_s[i].x + v4_s[i].y * v4_s[i].y + v4_s[i].z * v4_s[i].z;
			double ir = 1.0/sqrt(rsq);
			double ia = 2.0*ir-vsq/mu;
			dtgr[i] = 1.0 - 1.5 * mu * ia / c2;
		}
		else{
			dtgr[i] = 1.0;
		}
	}
	for(int i = N2; i < NN; ++i){
		x4_s[i].x = 0.0;
		x4_s[i].y = 0.0;
		x4_s[i].z = 0.0;
		x4_s[i].w = -1.0e-12;
		v4_s[i].x = 0.0;
		v4_s[i].y = 0.0;
		v4_s[i].z = 0.0;
		v4_s[i].w = 0.0;
		for(int l = 0; l < SLevels; ++l){
			rcritv_s[i + l * NN] = 0.0;
		}
		dtgr[i] = 1.0;
	}

	for(int i = 0; i < def_MaxColl; ++i){
		Colpairs_s[i].x = 0;
		Colpairs_s[i].y = 0;
		Coltime_s[i] = 10.0;
	}

	for(int tt = 0; tt < 10000; ++tt){

		for(int i = 0; i < N2; ++i){
			scalex_s[i].x = 1.0 / ((x4_s[i].x * x4_s[i].x) + 1.0e-20);
			scalex_s[i].y = 1.0 / ((x4_s[i].y * x4_s[i].y) + 1.0e-20);
			scalex_s[i].z = 1.0 / ((x4_s[i].z * x4_s[i].z) + 1.0e-20);

			scalev_s[i].x = 1.0 / ((v4_s[i].x * v4_s[i].x) + 1.0e-20);
			scalev_s[i].y = 1.0 / ((v4_s[i].y * v4_s[i].y) + 1.0e-20);
			scalev_s[i].z = 1.0 / ((v4_s[i].z * v4_s[i].z) + 1.0e-20);
		}

		for(int i = 0; i < N2; ++i){
			a0_s[i] = {0.0, 0.0, 0.0};
			for(int j = 0; j < N2; ++j){
				accEnc(x4_s[i], x4_s[j], a0_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
			}
			accEncSun(x4_s[i], a0_s[i], def_ksq * Msun * dtgr[i]);
		}

		int f = 1;

		for(int ff = 0; ff < 1e6; ++ff){
			for(int n = 1; n <= 8; ++n){
				dt2 = dt1 / (2.0 * n);
				dt22 = dt2 * 2.0;

				for(int i = 0; i < N2; ++i){
					xp_s[i].x = x4_s[i].x + (dt2 * dtgr[i] * v4_s[i].x);
					xp_s[i].y = x4_s[i].y + (dt2 * dtgr[i] * v4_s[i].y);
					xp_s[i].z = x4_s[i].z + (dt2 * dtgr[i] * v4_s[i].z);
					xp_s[i].w = x4_s[i].w;

					vp_s[i].x = v4_s[i].x + (dt2 * a0_s[i].x);
					vp_s[i].y = v4_s[i].y + (dt2 * a0_s[i].y);
					vp_s[i].z = v4_s[i].z + (dt2 * a0_s[i].z);
					vp_s[i].w = v4_s[i].w;
				}

				for(int i = 0; i < N2; ++i){
					a_s[i] = {0.0, 0.0, 0.0};
					for(int j = 0; j < N2; ++j){
						accEnc(xp_s[i], xp_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
					}
					accEncSun(xp_s[i], a_s[i], def_ksq * Msun * dtgr[i]);
				}

				for(int i = 0; i < N2; ++i){
					xt_s[i].x = x4_s[i].x + (dt22 * dtgr[i] * vp_s[i].x);
					xt_s[i].y = x4_s[i].y + (dt22 * dtgr[i] * vp_s[i].y);
					xt_s[i].z = x4_s[i].z + (dt22 * dtgr[i] * vp_s[i].z);
					xt_s[i].w = x4_s[i].w;

					vt_s[i].x = v4_s[i].x + (dt22 * a_s[i].x);
					vt_s[i].y = v4_s[i].y + (dt22 * a_s[i].y);
					vt_s[i].z = v4_s[i].z + (dt22 * a_s[i].z);
					vt_s[i].w = v4_s[i].w;
				}
				
				for(int m = 2; m <= n; ++m){

					for(int i = 0; i < N2; ++i){
						a_s[i] = {0.0, 0.0, 0.0};
						for(int j = 0; j < N2; ++j){
							accEnc(xt_s[i], xt_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
						}
						accEncSun(xt_s[i], a_s[i], def_ksq * Msun * dtgr[i]);
					}

					for(int i = 0; i < N2; ++i){
						xp_s[i].x += (dt22 * dtgr[i] * vt_s[i].x);
						xp_s[i].y += (dt22 * dtgr[i] * vt_s[i].y);
						xp_s[i].z += (dt22 * dtgr[i] * vt_s[i].z);

						vp_s[i].x += (dt22 * a_s[i].x);
						vp_s[i].y += (dt22 * a_s[i].y);
						vp_s[i].z += (dt22 * a_s[i].z);
					}

					for(int i = 0; i < N2; ++i){
						a_s[i] = {0.0, 0.0, 0.0};
						for(int j = 0; j < N2; ++j){
							accEnc(xp_s[i], xp_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
						}
						accEncSun(xp_s[i], a_s[i], def_ksq * Msun * dtgr[i]);
					}

					for(int i = 0; i < N2; ++i){
						xt_s[i].x += (dt22 * dtgr[i] * vp_s[i].x);
						xt_s[i].y += (dt22 * dtgr[i] * vp_s[i].y);
						xt_s[i].z += (dt22 * dtgr[i] * vp_s[i].z);

						vt_s[i].x += (dt22 * a_s[i].x);
						vt_s[i].y += (dt22 * a_s[i].y);
						vt_s[i].z += (dt22 * a_s[i].z);
					}
				}//end of m loop

				for(int i = 0; i < N2; ++i){
					a_s[i] = {0.0, 0.0, 0.0};
					for(int j = 0; j < N2; ++j){
						accEnc(xt_s[i], xt_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
					}
					accEncSun(xt_s[i], a_s[i], def_ksq * Msun * dtgr[i]);
				}

				for(int i = 0; i < N2; ++i){
					dx_s[i][n-1].x = 0.5 * (xt_s[i].x + (xp_s[i].x + (dt2 * dtgr[i] * vt_s[i].x)));
					dx_s[i][n-1].y = 0.5 * (xt_s[i].y + (xp_s[i].y + (dt2 * dtgr[i] * vt_s[i].y)));
					dx_s[i][n-1].z = 0.5 * (xt_s[i].z + (xp_s[i].z + (dt2 * dtgr[i] * vt_s[i].z)));

					dv_s[i][n-1].x = 0.5 * (vt_s[i].x + (vp_s[i].x + (dt2 * a_s[i].x)));
					dv_s[i][n-1].y = 0.5 * (vt_s[i].y + (vp_s[i].y + (dt2 * a_s[i].y)));
					dv_s[i][n-1].z = 0.5 * (vt_s[i].z + (vp_s[i].z + (dt2 * a_s[i].z)));	
//printf("dx %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, i, dx_s[i][n-1].x, dx_s[i][n-1].y, dx_s[i][n-1].z, dv_s[i][n-1].x, dv_s[i][n-1].y, dv_s[i][n-1].z);
				}

				error = 0.0;
				for(int i = 0; i < N2; ++i){
					for(int j = n-1; j >= 1; --j){
						double t0 = BSt0_c[(n-1) * 8 + (j-1)];
						double t1 = t0 * BSddt_c[j];
						double t2 = t0 * BSddt_c[n-1];

						dx_s[i][j-1].x = (t1 * dx_s[i][j].x) - (t2 * dx_s[i][j-1].x);
						dx_s[i][j-1].y = (t1 * dx_s[i][j].y) - (t2 * dx_s[i][j-1].y);
						dx_s[i][j-1].z = (t1 * dx_s[i][j].z) - (t2 * dx_s[i][j-1].z);

						dv_s[i][j-1].x = (t1 * dv_s[i][j].x) - (t2 * dv_s[i][j-1].x);
						dv_s[i][j-1].y = (t1 * dv_s[i][j].y) - (t2 * dv_s[i][j-1].y);
						dv_s[i][j-1].z = (t1 * dv_s[i][j].z) - (t2 * dv_s[i][j-1].z);
					}
					double errorx = (dx_s[i][0].x * dx_s[i][0].x) * scalex_s[i].x;
					double errorv = (dv_s[i][0].x * dv_s[i][0].x) * scalev_s[i].x;

					errorx = fmax(errorx, (dx_s[i][0].y * dx_s[i][0].y) * scalex_s[i].y);
					errorv = fmax(errorv, (dv_s[i][0].y * dv_s[i][0].y) * scalev_s[i].y);

					errorx = fmax(errorx, (dx_s[i][0].z * dx_s[i][0].z) * scalex_s[i].z);
					errorv = fmax(errorv, (dv_s[i][0].z * dv_s[i][0].z) * scalev_s[i].z);

					error = fmax(error, errorx);
					error = fmax(error, errorv);
//printf("%d %d %d %d %.20g %.20g %.20g %g %g %g %g | %g %g \n", i, tt, ff, n, error, dx_s[i][0].x, dx_s[i][0].y, dx_s[i][0].z, dv_s[i][0].x, dv_s[i][0].y, dv_s[i][0].z, t, dt1); 
	

				}
				Ncol_s[0] = 0;
				Coltime_s[0] = 10.0;

//printf("error %.20g %.20g %.20g %.20g\n", error, def_tol * def_tol, sgnt * dt1, def_dtmin);
				if(error < def_tol * def_tol || sgnt * dt1 < def_dtmin){

					for(int i = 0; i < N2; ++i){
						xt_s[i].x = dx_s[i][0].x;
						xt_s[i].y = dx_s[i][0].y;
						xt_s[i].z = dx_s[i][0].z;

						vt_s[i].x = dv_s[i][0].x;
						vt_s[i].y = dv_s[i][0].y;
						vt_s[i].z = dv_s[i][0].z;		

						for(int j = 1; j < n; ++j){
							xt_s[i].x += dx_s[i][j].x;
							xt_s[i].y += dx_s[i][j].y;
							xt_s[i].z += dx_s[i][j].z;

							vt_s[i].x += dv_s[i][j].x;
							vt_s[i].y += dv_s[i][j].y;
							vt_s[i].z += dv_s[i][j].z;
						}
//printf("xt %d %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g\n", tt, ff, n, i, xt_s[i].x, xt_s[i].y, xt_s[i].z, vt_s[i].x, vt_s[i].y, vt_s[i].z);
					}
					for(int i = 0; i < N2; ++i){
						for(int j = 0; j < N2; ++j){
							double delta = 1000.0;
							double enct = 100.0;
							double colt = 100.0;
							double rcrit = v4_s[i].w + v4_s[j].w;
							if((noColl == 1 || noColl == -1) && index_h[Encpairs2_h[start + i].x] == CollTshiftpairs_c[0].x && index_h[Encpairs2_h[start + j].x] == CollTshiftpairs_c[0].y){
								rcrit = v4_s[i].w * CollTshift_c[0] + v4_s[j].w * CollTshift_c[0];
							}
							if((noColl == 1 || noColl == -1) && index_h[Encpairs2_h[start + i].x] == CollTshiftpairs_c[0].y && index_h[Encpairs2_h[start + j].x] == CollTshiftpairs_c[0].x){
								rcrit = v4_s[i].w * CollTshift_c[0] + v4_s[j].w * CollTshift_c[0];
							}

							if(CollisionPrecision_c[0] < 0.0){
								//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
								rcrit *= (1.0 - CollisionPrecision_c[0]);	
							}
							if(Encpairs2_h[start + i].x > Encpairs2_h[start + j].x){
								delta = encounter1(xt_s[i], vt_s[i], x4_s[i], v4_s[i], xt_s[j], vt_s[j], x4_s[j], v4_s[j], rcrit, dt1 * dtgr[i], i, j, enct, colt, MinMass, noColl);
							}
							if((noColl == 1 || noColl == -1) && colt == 100.0){
								delta = 100.0;
							}
							if((noColl == 1 || noColl == -1) && colt == 200.0){
								noColl = 2;
								BSstop_h[0] = 3;
							}

//if( Encpairs2_h[start + i].x == 0 &&  Encpairs2_h[start + j].x == 1) printf("BSBEE %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %d %d %d\n", Encpairs2_h[start + i].x, Encpairs2_h[start + j].x, xt_s[i].w, xt_s[j].w, xt_s[i].x, xt_s[i].y, xt_s[i].z, xt_s[j].x, xt_s[j].y, xt_s[j].z, delta, rcrit*rcrit, colt, tt, ff, n);
							if(delta < rcrit*rcrit){
								int Ni;
								#pragma omp atomic capture
								Ni = Ncol_s[0]++;

								if(Ncol_s[0] >= def_MaxColl) Ni = def_MaxColl - 1;
//printf("EE1 %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %g %d\n", Encpairs2_h[start + i].x, Encpairs2_h[start + j].x, xt_s[i].w, xt_s[j].w, xt_s[i].x, xt_s[i].y, xt_s[i].z, xt_s[j].x, xt_s[j].y, xt_s[j].z, delta, rcrit*rcrit, colt, Ni);
								if(xt_s[i].w >= xt_s[j].w){
									Colpairs_s[Ni].x = i;
									Colpairs_s[Ni].y = j;
								}
								else{
									Colpairs_s[Ni].x = j;
									Colpairs_s[Ni].y = i;
								}
								Coltime_s[Ni] = colt;

								// *****************
								//dont group test particles
/*								if(xt_s[i].w == 0.0){
									Colpairs_s[Ni].x = i;
									Colpairs_s[Ni].y = i;
								}
								if(xt_s[j].w == 0.0){
									Colpairs_s[Ni].x = j;
									Colpairs_s[Ni].y = j;
								}
*/
								// *****************
							}

							if(WriteEncounters_c[0] > 0 && noColl == 0){
								double writeRadius = 0.0;
								//in scales of planetary Radius
								writeRadius = WriteEncountersRadius_c[0] * fmax(vt_s[i].w, vt_s[j].w);
//printf("Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, i, j);
								if(delta < writeRadius * writeRadius){

									if(enct > 0.0 && enct < 1.0){
										//ingnore encounters within the same particle cloud
										int indexi = Encpairs2_h[start + i].x;
										int indexj = Encpairs2_h[start + j].x;
										if(index_h[indexi] / WriteEncountersCloudSize_c[0] != index_h[indexj] / WriteEncountersCloudSize_c[0]){

//printf("Write Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, i, j);
											int ne;
											#pragma omp atomic capture
											ne = NWriteEnc_m[0]++;

											if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
											storeEncounters(xt_s, vt_s, i, j, indexi, indexj, index_h, ne, writeEnc_h, time + (t + dt1) / dayUnit, spin_h);
										}
									}
								}
							}
						}
					}

					double Coltime = 10.0;
					for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
						int i = Colpairs_s[c].x;
						int j = Colpairs_s[c].y;

						//Calculate real separation at the end of the time step
						double dx = xt_s[i].x - xt_s[j].x;
						double dy = xt_s[i].y - xt_s[j].y;
						double dz = xt_s[i].z - xt_s[j].z;
						double d = sqrt(dx * dx + dy * dy + dz * dz);
						double R = vt_s[i].w + vt_s[j].w;

						if((noColl == 1 || noColl == -1) && index_h[Encpairs2_h[start + i].x] == CollTshiftpairs_c[0].x && index_h[Encpairs2_h[start + j].x] == CollTshiftpairs_c[0].y){
							R = vt_s[i].w * CollTshift_c[0] + vt_s[j].w * CollTshift_c[0];
						}

						if(CollisionPrecision_c[0] < 0.0){
							//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
							R *= (1.0 - CollisionPrecision_c[0]);	
						}

						double dR = (R - d) / R;
						if(noColl == -1) dR = -dR;

//printf("dR %d %d %.20g %.20g %.20g %g\n", i, j, d, R, dR, Coltime_s[c]);
						if(dR > fabs(CollisionPrecision_c[0]) && d != 0.0){
							//bodies are already overlapping
							Coltime = fmin(Coltime_s[c], Coltime);
						}

					}
					Coltime_s[0] = Coltime;
//printf("ColtimeT %.20g %g %g %g %d %d %d %d %d\n", Coltime_s[0], t / dayUnit, dt1 / dayUnit, (1.0 - Coltime) * dt1, tt, ff, n, Ncol_s[0], Ncoll_m[0]);
					
					if(Coltime_s[0] == 10.0){
						for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;
							if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
								int nc = 0;
								if(noColl == 0 || ((noColl == 1 || noColl == -1) && index_h[Encpairs2_h[start + i].x] == CollTshiftpairs_c[0].x && index_h[Encpairs2_h[start + j].x] == CollTshiftpairs_c[0].y)){
									#pragma omp atomic capture
									nc = Ncoll_m[0]++;

									if(nc >= def_MaxColl) nc = def_MaxColl - 1;
									if(noColl == 1 || noColl == -1){
										noColl = 2;
										BSstop_h[0] = 3;
									}
								}
//printf("cTime coll BSB %g %g %g %.20g %d %d %d\n", time, t / dayUnit, dt /dayUnit, time + (t + dt1) / dayUnit, index_h[Encpairs2_h[start + i].x], index_h[Encpairs2_h[start + j].x], nc);
								collide(random, xt_s, vt_s, i, j, Encpairs2_h[start + i].x, Encpairs2_h[start + j].x, Msun, U_h, test, index_h, nc, Coll_h, time + (t + dt1) / dayUnit, spin_h, love_h, createFlag_h, rcritv_s, rcrit_h, NN, NconstT, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, SLevels, noColl);
							}
						}

						t += dt1;
						if(n >= 8) dt1 *= 0.55;
						if(n < 7) dt1 *= 1.3;
						if(sgnt * dt1 > sgnt * dt) dt1 = dt;
						if(sgnt * (t+dt1) > sgnt * dt) dt1 = dt - t;
						if(sgnt * dt1 < def_dtmin) dt1 = sgnt * def_dtmin;

						for(int i = 0; i < N2; ++i){
							x4_s[i] = xt_s[i];
							v4_s[i] = vt_s[i];
//printf("update %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.10g %g %g %d %d %d\n", i, idx, x4_s[i].x, x4_s[i].y, x4_s[i].z, v4_s[i].x, v4_s[i].y, v4_s[i].z, x4_s[i].w, v4_s[i].w, rcritv_s[i], t / dayUnit, dt1 / dayUnit, tt, ff, n);
						}
					}
					else{
						dt1 *= Coltime_s[0];
//printf("New Coltime %g %g\n", dt1, Coltime_s[0]);
					}
					f = 0;

					break;
				}
				if(BSstop_h[0] == 3){
//printf("Stop BSB\n");
					return;
				}
			}//end of n loop
			if(f == 0) break;
			dt1 *= 0.5;
		}//end of ff loop
		if(sgnt * t >= sgnt * dt){
			break;
		}

	}//end of tt loop

	for(int i = 0; i < N2; ++i){
//if(x4_s[i].w <= 0){
		int idi = Encpairs2_h[start + i].x;
		x4_h[idi] = x4_s[i]; 
		v4_h[idi] = v4_s[i];
		for(int l = 0; l < SLevels; ++l){  
			rcritv_h[idi + l * NconstT] = rcritv_s[i + l * NN];
		}
//}
//printf("BS %d %.40g %.40g %.40g %.40g %.40g %.40g %.20g\n", i, x4_s[i].x, x4_s[i].y, x4_s[i].z, v4_s[i].x, v4_s[i].y, v4_s[i].z, time + t/dayUnit);
	}
//printf("BSB end   %d %d\n", NN, idx);
}
