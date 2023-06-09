
template <int NN>
void BSA_cpu(int *random_h, double4 *x4_h, double4 *v4_h, double4 *xold_h, double4 *vold_h, double *rcrit_h, double *rcritv_h, int *index_h, double4 *spin_h, double3 *love_h, int *createFlag_h, int2 *Encpairs_h, int2 *Encpairs2_h, const double dt, const double Msun, double *U_h, const int st, const int NT, const int NconstT, const int NencMax, int *BSstop_h, int *Ncoll_m, double *Coll_h, const double time, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *enccount_h, unsigned long long *aecountT_h, unsigned long long *enccountT_h, int *NWriteEnc_m, double *writeEnc_h, const int UseGR, const double MinMass, const int UseTestParticles, const int SLevels, int noColl, int idx){


	int random = 0;
//printf("BSA %d %g\n", idx, StopMinMass_c[0]);

	if((noColl == 1 || noColl == -1) && BSstop_h[0] == 3){
//if(idx == 0)	printf("Stop BSA b\n");
		return;
	}

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
	int sgnt = 1;

	double3 scalex_s[NN];
	double3 scalev_s[NN];

	double error = 0.0;
	double test;
	int si = Encpairs2_h[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_h[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_h[NT + si].y;

	if(dt < 0.0){
		sgnt = -1;
	}

//printf("BS %d %d %d %d %d\n", idx, st, si, N2, NT);


	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
		x4_s[i] = xold_h[idi];
		v4_s[i] = vold_h[idi];
		for(int l = 0; l < SLevels; ++l){
			rcritv_s[i + l * NN] = rcritv_h[idi + l * NconstT];
		}

//printf("BSA2 %d %d %d %d %d %d %d\n", idx, i, st, idi, index_h[idi], N2, Ne);
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
			scalex_s[i].x = 1.0 / (x4_s[i].x * x4_s[i].x + 1.0e-20);
			scalex_s[i].y = 1.0 / (x4_s[i].y * x4_s[i].y + 1.0e-20);
			scalex_s[i].z = 1.0 / (x4_s[i].z * x4_s[i].z + 1.0e-20);

			scalev_s[i].x = 1.0 / (v4_s[i].x * v4_s[i].x + 1.0e-20);
			scalev_s[i].y = 1.0 / (v4_s[i].y * v4_s[i].y + 1.0e-20);
			scalev_s[i].z = 1.0 / (v4_s[i].z * v4_s[i].z + 1.0e-20);
		}

		for(int i = 0; i < N2; ++i){
			a0_s[i] = {0.0, 0.0, 0.0};
			int idi = Encpairs2_h[start + i].x;
			int Ne = Encpairs_h[idi].y;
			for(int ii = 0; ii < Ne; ++ii){
				int jg = Encpairs_h[idi * NencMax + ii].x;
				int j = Encpairs_h[NT + jg].y;
				accEnc(x4_s[i], x4_s[j], a0_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
			}
			if(Ne >= 0){
				accEncSun(x4_s[i], a0_s[i], def_ksq * Msun * dtgr[i]);
			}
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
					int idi = Encpairs2_h[start + i].x;
					int Ne = Encpairs_h[idi].y;
					for(int ii = 0; ii < Ne; ++ii){
						int jg = Encpairs_h[idi * NencMax + ii].x;
						int j = Encpairs_h[NT + jg].y;
						accEnc(xp_s[i], xp_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
					}
					accEncSun(xp_s[i], a_s[i], def_ksq * Msun * dtgr[i]);

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
						int idi = Encpairs2_h[start + i].x;
						int Ne = Encpairs_h[idi].y;
						for(int ii = 0; ii < Ne; ++ii){
							int jg = Encpairs_h[idi * NencMax + ii].x;
							int j = Encpairs_h[NT + jg].y;
							accEnc(xt_s[i], xt_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
						}
						accEncSun(xt_s[i], a_s[i], def_ksq * Msun * dtgr[i]);

						xp_s[i].x += (dt22 * dtgr[i] * vt_s[i].x);
						xp_s[i].y += (dt22 * dtgr[i] * vt_s[i].y);
						xp_s[i].z += (dt22 * dtgr[i] * vt_s[i].z);

						vp_s[i].x += (dt22 * a_s[i].x);
						vp_s[i].y += (dt22 * a_s[i].y);
						vp_s[i].z += (dt22 * a_s[i].z);
					}
			
					for(int i = 0; i < N2; ++i){
						a_s[i] = {0.0, 0.0, 0.0};
						int idi = Encpairs2_h[start + i].x;
						int Ne = Encpairs_h[idi].y;
						for(int ii = 0; ii < Ne; ++ii){
							int jg = Encpairs_h[idi * NencMax + ii].x;
							int j = Encpairs_h[NT + jg].y;
							accEnc(xp_s[i], xp_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
						}
						accEncSun(xp_s[i], a_s[i], def_ksq * Msun * dtgr[i]);

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
					int idi = Encpairs2_h[start + i].x;
					int Ne = Encpairs_h[idi].y;
					for(int ii = 0; ii < Ne; ++ii){
						int jg = Encpairs_h[idi * NencMax + ii].x;
						int j = Encpairs_h[NT + jg].y;
						accEnc(xt_s[i], xt_s[j], a_s[i], rcritv_s, test, i, j, NN, MinMass, UseTestParticles, SLevels);
					}

					accEncSun(xt_s[i], a_s[i], def_ksq * Msun * dtgr[i]);

					xp_s[i].x += (dt2 * dtgr[i] * vt_s[i].x);
					xp_s[i].y += (dt2 * dtgr[i] * vt_s[i].y);
					xp_s[i].z += (dt2 * dtgr[i] * vt_s[i].z);

					vp_s[i].x += (dt2 * a_s[i].x);
					vp_s[i].y += (dt2 * a_s[i].y);
					vp_s[i].z += (dt2 * a_s[i].z);
				}


				for(int i = 0; i < N2; ++i){
					dx_s[i][n-1].x = 0.5 * (xt_s[i].x + xp_s[i].x);
					dx_s[i][n-1].y = 0.5 * (xt_s[i].y + xp_s[i].y);
					dx_s[i][n-1].z = 0.5 * (xt_s[i].z + xp_s[i].z);

					dv_s[i][n-1].x = 0.5 * (vt_s[i].x + vp_s[i].x);
					dv_s[i][n-1].y = 0.5 * (vt_s[i].y + vp_s[i].y);
					dv_s[i][n-1].z = 0.5 * (vt_s[i].z + vp_s[i].z);
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

				}
				Ncol_s[0] = 0;
				Coltime_s[0] = 10.0;

				if(error < def_tol * def_tol || sgnt * dt1 < def_dtmin){
//printf("tt %d %d %d %g\n", tt, ff, n, Coltime_s[0]);
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

					}
	
					for(int i = 0; i < N2; ++i){
						int idi = Encpairs2_h[start + i].x;
						int Ne = Encpairs_h[idi].y;
						for(int ii = 0; ii < Ne; ++ii){
							double delta = 100.0;
							double enct = 100.0;
							double colt = 100.0;
							int jg = Encpairs_h[idi * NencMax + ii].x;
							int j = Encpairs_h[NT + jg].y;
							double rcrit = v4_s[i].w + v4_s[j].w;
							if((noColl == 1 || noColl == -1) && index_h[idi] == CollTshiftpairs_c[0].x && index_h[jg] == CollTshiftpairs_c[0].y){
								rcrit = v4_s[i].w * CollTshift_c[0] + v4_s[j].w * CollTshift_c[0];
							}
							if((noColl == 1 || noColl == -1) && index_h[idi] == CollTshiftpairs_c[0].y && index_h[jg] == CollTshiftpairs_c[0].x){
								rcrit = v4_s[i].w * CollTshift_c[0] + v4_s[j].w * CollTshift_c[0];
							}
							if(CollisionPrecision_c[0] < 0.0){
								//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
								rcrit *= (1.0 - CollisionPrecision_c[0]);	
							}
//printf("%d %d %d %d %d\n", Encpairs2_h[start + i].x, Encpairs2_h[start + j].x, jg, j, idi);
							if(idi > jg){
								delta = encounter1(xt_s[i], vt_s[i], x4_s[i], v4_s[i], xt_s[j], vt_s[j], x4_s[j], v4_s[j], rcrit, dt1 * dtgr[i], i, j, enct, colt, MinMass, noColl);
							}
							if((noColl == 1 || noColl == -1) && colt == 100.0){
								delta = 100.0;
							}
							if((noColl == 1 || noColl == -1) && colt == 200.0){
								noColl = 2;
								BSstop_h[0] = 3;
							}
							if(delta < rcrit*rcrit){
								int Ni = Ncol_s[0]++;
								if(Ncol_s[0] >= def_MaxColl) Ni = def_MaxColl - 1;
	//printf("EE1 %d %d %d %d | %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %d %d %g %d\n", idi, jg, index_h[idi], index_h[jg], xt_s[i].w, xt_s[j].w, xt_s[i].x, xt_s[i].y, xt_s[i].z, xt_s[j].x, xt_s[j].y, xt_s[j].z, delta, rcrit*rcrit, f, n, colt, Ni);

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
	/*							if(xt_s[i].w == 0.0){
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
								if(delta < writeRadius * writeRadius){

									if(enct > 0.0 && enct < 1.0){
										//ingnore encounters within the same particle cloud
										if(index_h[idi] / WriteEncountersCloudSize_c[0] != index_h[jg] / WriteEncountersCloudSize_c[0]){

	//printf("Write Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, ii, jg);
											int ne = NWriteEnc_m[0]++;
											if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
											storeEncounters(xt_s, vt_s, i, j, idi, jg, index_h, ne, writeEnc_h, time + (t + dt1) / dayUnit, spin_h);
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

//printf("dRA %d %d %.20g %.20g %.20g\n", i, j, d, R, dR);
						if(dR > fabs(CollisionPrecision_c[0]) && d != 0.0){
						//bodies are already overlapping
							Coltime = fmin(Coltime_s[c], Coltime);
						}


					}
					Coltime_s[0] = Coltime;
//printf("ColtimeT %.20g %g %g %g %d %d %d %d %d\n", Coltime, t / dayUnit, dt1 / dayUnit, (1.0 - Coltime) * dt1, tt, ff, n, Ncol_s[0], Ncoll_m[0]);
					if(Coltime_s[0] == 10.0){
						for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
							int i = Colpairs_s[c].x;
							int j = Colpairs_s[c].y;
							if(xt_s[i].w >= 0 && xt_s[j].w >= 0){
								int nc = 0;
								if(noColl == 0 || ((noColl == 1 || noColl == -1) && index_h[Encpairs2_h[start + i].x] == CollTshiftpairs_c[0].x && index_h[Encpairs2_h[start + j].x] == CollTshiftpairs_c[0].y)){
									nc = Ncoll_m[0]++;
									if(nc >= def_MaxColl) nc = def_MaxColl - 1;
									if(noColl == 1 || noColl == -1){
										noColl = 2;
										BSstop_h[0] = 3;
									}
								}
//printf("cTime coll BSA %g %g %g %.20g %d %d %d\n", time, t / dayUnit, dt / dayUnit, time + (t + dt1) / dayUnit, index_h[Encpairs2_h[start + i].x], index_h[Encpairs2_h[start + j].x], nc);
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
//printf("update %d %d %d %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.10g %g %g %d %d %d\n", i, idx, idi, x4_s[i].x, x4_s[i].y, x4_s[i].z, v4_s[i].x, v4_s[i].y, v4_s[i].z, x4_s[i].w, v4_s[i].w, rcritv_s[i], t / dayUnit, dt1 / dayUnit, tt, ff, n);
						}
					}
					else{
						dt1 *= Coltime_s[0];
					}
					f = 0;

					break;
				}
				if(BSstop_h[0] == 3){
//printf("Stop BSA\n");
					return;
				}
			} //end of n loop
			if(f == 0) break;
			dt1 *= 0.5;
//printf("continue %d %g %g %d %d\n", idx, t, dt1, tt, ff);
		}//end of ff loop
		if(sgnt * t >= sgnt * dt){
			break;
		}
//printf("not finished %d\n", idx);

	}//end of tt loop
	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
//if(xt_s[i].w <= 0){
		x4_h[idi] = xt_s[i];
		v4_h[idi] = vt_s[i];
		for(int l = 0; l < SLevels; ++l){  
			rcritv_h[idi + l * NconstT] = rcritv_s[i + l * NN];
		}
//}
	}
}



template <int E>
void BSAcc_cpu(double4 *x4_h, double4 *v4_h, double4 *xA_h, double4 *vA_h, double4 *xB_h, double4 *vB_h, double *rcritv_h, int2 *Encpairs_h, int2 *Encpairs2_h, double *dt1_h, double *dtgr_h, const double Msun, const int st, const int NT, const int NconstT, const int NencMax, int *BSAstop_h, double *Coltime_h, const int n, const int f, const double MinMass, const int UseTestParticles, const double dt, const int SLevels, int noColl, int idx){


	int si = Encpairs2_h[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_h[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_h[NT + si].y;
//if(n == 1 && E == 0) printf("BS %d %d %d %d %d\n", idx, st, si, N2, NT);
	#pragma omp parallel for
	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
		int Ne;
		if(E == 0 && n == 1){
			if(f == 0){
				Ne = Encpairs_h[idi].y; //number of pairs
				Encpairs_h[idi + 7 * NT].y = Ne;
				Encpairs_h[idi + 6 * NT].y = Ne;
			}
			else{
				Ne = Encpairs_h[idi + 7 * NT].y; //number of pairs
				Encpairs_h[idi + 6 * NT].y = Ne;
			}
		}
		else Ne = Encpairs_h[idi + 6 * NT].y; //number of pairs
		if(Ne >= 0){
			double dt1 = dt1_h[idi];
			double dt2 = dt1 / (2.0 * n);
			double dtgr = dtgr_h[idi];
			if(E == 1 || E == 2) dt2 *= 2.0;

			double4 xAi = xA_h[idi];
			double4 vAi = vA_h[idi];

			double3 a = {0.0, 0.0, 0.0};
			double test;

			for(int ii = 0; ii < Ne; ++ii){	
				int j = Encpairs_h[idi * NencMax + ii].x;
//if(n == 1 && E == 0)  printf("%d %d %d %d %d %d %d %d %g\n", idx, i, idi, Ne, N2, start, j, n, dt1);
				double4 xAj = xA_h[j];

				accEnc(xAi, xAj, a, rcritv_h, test, idi, j, NconstT, MinMass, UseTestParticles, SLevels);		
		
			}	
			accEncSun(xAi, a, def_ksq * Msun * dtgr);
			double4 x4B;
			double4 v4B;

			if(E == 0){
				//here xA = x4, and xB = xp
				x4B.x = xAi.x + (dt2 * dtgr * vAi.x);
				x4B.y = xAi.y + (dt2 * dtgr * vAi.y);
				x4B.z = xAi.z + (dt2 * dtgr * vAi.z);
				x4B.w = xAi.w;
				v4B.x = vAi.x + (dt2 * a.x);
				v4B.y = vAi.y + (dt2 * a.y);
				v4B.z = vAi.z + (dt2 * a.z);
				v4B.w = vAi.w;
				Encpairs_h[si + 5 * NT].y = 0; //set accept condition
			}
			if(E == 1){
				//here xA = xp, and xB = xt
				double4 x4i = x4_h[idi];
				double4 v4i = v4_h[idi];
				x4B.x = x4i.x + (dt2 * dtgr * vAi.x);
				x4B.y = x4i.y + (dt2 * dtgr * vAi.y);
				x4B.z = x4i.z + (dt2 * dtgr * vAi.z);
				x4B.w = x4i.w;

				v4B.x = v4i.x + (dt2 * a.x);
				v4B.y = v4i.y + (dt2 * a.y);
				v4B.z = v4i.z + (dt2 * a.z);
				v4B.w = v4i.w;
			}
			if(E == 2 || E == 3){
				x4B = xB_h[idi];
				v4B = vB_h[idi];
				x4B.x += (dt2 * dtgr * vAi.x);
				x4B.y += (dt2 * dtgr * vAi.y);
				x4B.z += (dt2 * dtgr * vAi.z);

				v4B.x += (dt2 * a.x);
				v4B.y += (dt2 * a.y);
				v4B.z += (dt2 * a.z);
			}

			xB_h[idi] = x4B;
			vB_h[idi] = v4B;
//if(idi == 1 && E == 0 && n == 1) printf("xB %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xB_h[idi].x, xB_h[idi].y, xB_h[idi].z, vB_h[idi].x, vB_h[idi].y, vB_h[idi].z, n);
		}		
	}
	if(E == 0){
		BSAstop_h[0] = 1;
		Coltime_h[0] = 10.0;
	}
}

void BSError_cpu(double4 *x4_h, double4 *v4_h, double4 *xp_h, double4 *vp_h, double4 *xt_h, double4 *vt_h, double3 *dx_h, double3 *dv_h, int2 *Encpairs_h, int2 *Encpairs2_h, const int st, const int NT, const int n, const int f, int idx){

	int si = Encpairs2_h[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_h[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_h[NT + si].y;

	#pragma omp parallel for
	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
		int Ne = Encpairs_h[idi + 6 * NT].y; //number of pairs
		if(Ne >= 0){
			double4 x;
			double4 v;
			double3 dxj; //this is dx[n-1]
			double3 dvj;
			x = x4_h[idi];
			v = v4_h[idi];
	
			double3 scalex;
			double3 scalev;
			scalex.x = 1.0 / (x.x * x.x + 1.0e-20);
			scalex.y = 1.0 / (x.y * x.y + 1.0e-20);
			scalex.z = 1.0 / (x.z * x.z + 1.0e-20);
			scalev.x = 1.0 / (v.x * v.x + 1.0e-20);
			scalev.y = 1.0 / (v.y * v.y + 1.0e-20);
			scalev.z = 1.0 / (v.z * v.z + 1.0e-20);

//if(idi == 1) printf("xp %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xp_h[idi].x, xp_h[idi].y, xp_h[idi].z, vp_h[idi].x, vp_h[idi].y, vp_h[idi].z, n);
//if(idi == 1) printf("xt %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, xt_h[idi].x, xt_h[idi].y, xt_h[idi].z, vt_h[idi].x, vt_h[idi].y, vt_h[idi].z, n);


			dxj.x = 0.5 * (xt_h[idi].x + xp_h[idi].x);
			dx_h[(n - 1) * NT + idi].x = dxj.x;
			dxj.y = 0.5 * (xt_h[idi].y + xp_h[idi].y);
			dx_h[(n - 1) * NT + idi].y = dxj.y;
			dxj.z = 0.5 * (xt_h[idi].z + xp_h[idi].z);
			dx_h[(n - 1) * NT + idi].z = dxj.z;
			dvj.x = 0.5 * (vt_h[idi].x + vp_h[idi].x);
			dv_h[(n - 1) * NT + idi].x = dvj.x;
			dvj.y = 0.5 * (vt_h[idi].y + vp_h[idi].y);
			dv_h[(n - 1) * NT + idi].y = dvj.y;
			dvj.z = 0.5 * (vt_h[idi].z + vp_h[idi].z);
			dv_h[(n - 1) * NT + idi].z = dvj.z;
//if(idi == 1) printf("dxj %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z, n);

			double dj1;

			for(int j = n - 1; j >= 1; --j){

				double t0 = BSt0_c[(n-1) * 8 + (j-1)];
				double t1 = t0 * BSddt_c[j];
				double t2 = t0 * BSddt_c[n-1];
				
				dj1 = dx_h[(j - 1) * NT + idi].x;
				dxj.x = (t1 * dxj.x) - (t2 * dj1);
				dx_h[(j - 1) * NT + idi].x = dxj.x;
				dj1 = dx_h[(j - 1) * NT + idi].y;
				dxj.y = (t1 * dxj.y) - (t2 * dj1);
				dx_h[(j - 1) * NT + idi].y = dxj.y;
				dj1 = dx_h[(j - 1) * NT + idi].z;
				dxj.z = (t1 * dxj.z) - (t2 * dj1);
				dx_h[(j - 1) * NT + idi].z = dxj.z;
				dj1 = dv_h[(j - 1) * NT + idi].x;
				dvj.x = (t1 * dvj.x) - (t2 * dj1);
				dv_h[(j - 1) * NT + idi].x = dvj.x;
				dj1 = dv_h[(j - 1) * NT + idi].y;
				dvj.y = (t1 * dvj.y) - (t2 * dj1);
				dv_h[(j - 1) * NT + idi].y = dvj.y;
				dj1 = dv_h[(j - 1) * NT + idi].z;
				dvj.z = (t1 * dvj.z) - (t2 * dj1);
				dv_h[(j - 1) * NT + idi].z = dvj.z;
			}
			double error = dxj.x * dxj.x * scalex.x;
			error = fmax(error, dxj.y * dxj.y * scalex.y);
			error = fmax(error, dxj.z * dxj.z * scalex.z);
			error = fmax(error, dvj.x * dvj.x * scalev.x);
			error = fmax(error, dvj.y * dvj.y * scalev.y);
			error = fmax(error, dvj.z * dvj.z * scalev.z);
//if(idi == 1) printf("dx %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, dxj.x, dxj.y, dxj.z, dvj.x, dvj.y, dvj.z, n);
//if(idi == 1) printf("scale %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idx, idi, scalex.x, scalex.y, scalex.z, scalev.x, scalev.y, scalev.z, n);
			if(error >= def_tol * def_tol){
				//dont accept BS step
				Encpairs_h[si + 5 * NT].y = 1; //Accept
//if(idi == 3534) printf("error %d %d %d %d %d %.20g %.20g %.20g %.20g\n", 0, f, n, id, idi, error, dxj.x, dxj.y, dxj.z);
			}
		}
	}
}


void BSAccept_cpu(double4 *xt_h, double4 *vt_h, double3 *dx_h, double3 *dv_h, int2 *Encpairs_h, int2 *Encpairs2_h, double *dt1_h, const int st, const int NT, const int n, int idx){


	int si = Encpairs2_h[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_h[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_h[NT + si].y;

	#pragma omp parallel for
	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
		int Ne = Encpairs_h[idi + 6 * NT].y; //group index
		if(Ne >= 0){
			int accept = Encpairs_h[si + 5 * NT].y;
			double3 xt;
			double3 vt;
			double dt1 = dt1_h[idi];
			int sgnt = 1;
			if(dt1 < 0.0) sgnt = -1;
			if(accept == 0 || sgnt * dt1 < def_dtmin){
				xt.x = dx_h[idi].x;
				xt.y = dx_h[idi].y;
				xt.z = dx_h[idi].z;
				vt.x = dv_h[idi].x;
				vt.y = dv_h[idi].y;
				vt.z = dv_h[idi].z;
//if(idi == 3159) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, 0, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
//if(idi == 566) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, 0, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);

				for(int j = 1; j < n; ++j){
					xt.x += dx_h[j * NT + idi].x;
					xt.y += dx_h[j * NT + idi].y;
					xt.z += dx_h[j * NT + idi].z;
					vt.x += dv_h[j * NT + idi].x;
					vt.y += dv_h[j * NT + idi].y;
					vt.z += dv_h[j * NT + idi].z;
//if(idi == 3159) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, j, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
//if(idi == 566) printf("xt %d %d %.20g %.20g %.20g %.20g %.20g %.20g %d\n", idi, j, xt.x, xt.y, xt.z, vt.x, vt.y, vt.z, n);
				}
				xt_h[idi].x = xt.x;
				xt_h[idi].y = xt.y;
				xt_h[idi].z = xt.z;
				vt_h[idi].x = vt.x;
				vt_h[idi].y = vt.y;
				vt_h[idi].z = vt.z;
			}
		}
	}
}


void BSUpdate_cpu(int *random_h, double4 *xold_h, double4 *vold_h, double4 *x4_h, double4 *v4_h, double4 *xt_h, double4 *vt_h, double *rcrit_h, double *rcritv_h, int *index_h, double4 *spin_h, double3 *love_h, int *createFlag_h, int2 *Encpairs_h, int2 *Encpairs2_h, int *BSAstop_h, double *dt1_h, double *t1_h, const double dt, const double Msun, double *U_h, const int st, const int NT, const int NconstT, const int f, const int n, const int NencMax, int *Ncoll_m, double *Coll_h, const double time, float4 *aelimits_h, unsigned int *aecount_h, unsigned int *enccount_h, unsigned long long *aecountT_h, unsigned long long *enccountT_h, int *NWriteEnc_m, double *writeEnc_h, double *dtgr_h, double *Coltime_h, const double MinMass, const int UseTestParticles, const int SLevels, int noColl, int idx){

	int random = 0;

	int si = Encpairs2_h[ (st+2) * NT + idx].y;
	int N2 = Encpairs2_h[si].y; //Number of bodies in  current BS simulation
	int start = Encpairs2_h[NT + si].y;

	int Ncol_s[1];
	int2 Colpairs_s[def_MaxColl];
	double Coltime_s[def_MaxColl];
	double test;
	Ncol_s[0] = 0;

	#pragma omp parallel for
	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
		int Ne = Encpairs_h[idi + 6 * NT].y; //number of pairs
		if(Ne >= 0){
			int accept = Encpairs_h[si + 5 * NT].y;
			double dt1 = dt1_h[idi];
			double t1 = t1_h[idi];
			double dtgr = dtgr_h[idi];
			int sgnt = 1;
			if(dt < 0.0) sgnt= -1;

			if(accept == 0 || sgnt * dt1 < def_dtmin){
//if(id == 0) printf("acceptA %d %d %d %d %d %d %.20g %.20g %.20g\n", idx, idi, si, n, accept, sgnt, dt, t1, dt1);

				for(int ii = 0; ii < Ne; ++ii){
					double delta = 100.0;
					double enct = 100.0;
					double colt = 100.0;
					volatile int j = Encpairs_h[idi * NencMax + ii].x;
					double rcrit = vold_h[idi].w + vold_h[j].w;
					if((noColl == 1 || noColl == -1) && index_h[idi] == CollTshiftpairs_c[0].x && index_h[j] == CollTshiftpairs_c[0].y){
						rcrit = vold_h[idi].w * CollTshift_c[0] + vold_h[j].w * CollTshift_c[0];
					}
					if((noColl == 1 || noColl == -1) && index_h[idi] == CollTshiftpairs_c[0].y && index_h[j] == CollTshiftpairs_c[0].x){
						rcrit = vold_h[idi].w * CollTshift_c[0] + vold_h[j].w * CollTshift_c[0];
					}
					if(CollisionPrecision_c[0] < 0.0){
						//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
						rcrit *= (1.0 - CollisionPrecision_c[0]);	
					}

					if(idi > j){
						delta = encounter1(xt_h[idi], vt_h[idi], xold_h[idi], vold_h[idi], xt_h[j], vt_h[j], xold_h[j], vold_h[j], rcrit, dt1 * dtgr, idi, j, enct, colt, MinMass, noColl);
					}
//if(idi == 12888 && j == 11191) printf("delta %g %g %g\n", enct, colt, delta);
					if((noColl == 1 || noColl == -1) && colt == 100.0){
						delta = 100.0;
					}
					if((noColl == 1 || noColl == -1) && colt == 200.0){
						noColl = 2;
						BSAstop_h[0] = 3;
					}
					if(delta < rcrit*rcrit){
						int Ni = Ncol_s[0]++;
						if(Ncol_s[0] >= def_MaxColl) Ni = def_MaxColl - 1;
//printf("EE1 %d %d %d %d | %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %.20g %d %d %g %d\n", idi, j, index_h[idi], index_h[j], xt_h[idi].w, xt_h[j].w, xt_h[idi].x, xt_h[idi].y, xt_h[idi].z, xt_h[j].x, xt_h[j].y, xt_h[j].z, delta, rcrit*rcrit, f, n, colt, Ni);
						if(xt_h[idi].w >= xt_h[j].w){
							Colpairs_s[Ni].x = idi;
							Colpairs_s[Ni].y = j;
						}
						else{
							Colpairs_s[Ni].x = j;
							Colpairs_s[Ni].y = idi;
						}
						Coltime_s[Ni] = colt;
			
						// *****************
						//dont group test particles
/*						if(xt_h[idi].w == 0.0){
							Colpairs_s[Ni].x = idi;
							Colpairs_s[Ni].y = idi;
						}
						if(xt_h[j].w == 0.0){
							Colpairs_s[Ni].x = j;
							Colpairs_s[Ni].y = j;
						}
*/
						// *****************
					}

					if(WriteEncounters_c[0] > 0 && noColl == 0){
						double writeRadius = 0.0;
						//in scales of planetary Radius
						writeRadius = WriteEncountersRadius_c[0] * fmax(vt_h[idi].w, vt_h[j].w);
						if(delta < writeRadius * writeRadius){

							if(enct > 0.0 && enct < 1.0){
								//ingnore encounters within the same particle cloud
								if(index_h[idi] / WriteEncountersCloudSize_c[0] != index_h[j] / WriteEncountersCloudSize_c[0]){
//printf("Write Enc %g %g %g %g %g %d %d\n", (t + dt1) / dayUnit, writeRadius, sqrt(delta), enct, colt, idi, j);
									int ne = NWriteEnc_m[0]++;
									if(ne >= def_MaxWriteEnc - 1) ne = def_MaxWriteEnc - 1;
									storeEncounters(xt_h, vt_h, idi, j, idi, j, index_h, ne, writeEnc_h, time + (t1 + dt1) / dayUnit, spin_h);
								}
							}
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
		double dx = xt_h[i].x - xt_h[j].x;
		double dy = xt_h[i].y - xt_h[j].y;
		double dz = xt_h[i].z - xt_h[j].z;
		double d = sqrt(dx * dx + dy * dy + dz * dz);
		double R = vt_h[i].w + vt_h[j].w;

		if((noColl == 1 || noColl == -1) && index_h[i] == CollTshiftpairs_c[0].x && index_h[j] == CollTshiftpairs_c[0].y){
			R = vt_h[i].w * CollTshift_c[0] + vt_h[j].w * CollTshift_c[0];
		}
		if(CollisionPrecision_c[0] < 0.0){
			//do not overlap bodies when collision, increase therefore radius slightly, R + R * precision
			R *= (1.0 - CollisionPrecision_c[0]);	
		}

		double dR = (R - d) / R;

		if(noColl == -1) dR = -dR;

//printf("dRm %d %d %.20g %.20g %.20g | noColl %d\n", i, j, d, R, dR, noColl);
		if(dR > fabs(CollisionPrecision_c[0]) && d != 0.0){
			//bodies are already overlapping
			Coltime = fmin(Coltime_s[c], Coltime);
		}
	}
	Coltime_h[0] = Coltime;
//printf("ColtimeT BSAm %.20g %g %g %g %d %d %d\n", Coltime, t1_h[0] / dayUnit, dt1_h[0] / dayUnit, (1.0 - Coltime) * dt1_h[0], n, Ncol_s[0], Ncoll_m[0]);


	for(int c = 0; c < min(Ncol_s[0], def_MaxColl); ++c){
		int i = Colpairs_s[c].x;
		int j = Colpairs_s[c].y;
		double t1 = t1_h[i];
		if(Coltime_h[0] == 10.0){
			if(xt_h[i].w >= 0 && xt_h[j].w >= 0){
				int nc = 0;
				if(noColl == 0 || ((noColl == 1 || noColl == -1) && index_h[i] == CollTshiftpairs_c[0].x && index_h[j] == CollTshiftpairs_c[0].y)){
					nc = Ncoll_m[0]++;
					if(nc >= def_MaxColl) nc = def_MaxColl - 1;
					if(noColl == 1 || noColl == -1){
						noColl = 2;
						BSAstop_h[0] = 3;
					}
				}
//printf("cTime coll BSAm %g %g %g %.20g %d %d %d | noColl %d\n", time, t1_h[i] / dayUnit, dt /dayUnit, time + (t1_h[i] + dt1_h[i]) / dayUnit, index_h[i], index_h[j], nc, noColl);
				collide(random, xt_h, vt_h, i, j, i, j, Msun, U_h, test, index_h, nc, Coll_h, time + (t1 + dt1_h[i]) / dayUnit, spin_h, love_h, createFlag_h, rcritv_h, rcrit_h, NconstT, NconstT, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, SLevels, noColl);
			}
		}
	}

	for(int i = 0; i < N2; ++i){
		int idi = Encpairs2_h[start + i].x;
		int Ne = Encpairs_h[idi + 7 * NT].y; //number of pairs
		int Ne1 = Encpairs_h[idi + 6 * NT].y; //number of pairs
//if(idi == 5942 || idi == 3472) printf("B %d %d %d %d\n", id, idi, Ne, Ne1);
		if(Ne >= 0){
			int accept = Encpairs_h[si + 5 * NT].y;
			volatile double dt1 = dt1_h[idi];
			volatile double t1 = t1_h[idi];
			int sgnt = 1;
			if(dt < 0.0) sgnt= -1;
//if(idi == 5942 || idi == 3472) printf("C %d %d %g %g %g %g %d %d %d\n", id, idi, Coltime_h[0], t1_h[idi], dt1_h[idi], 1.0 - CollisionPrecision_c[0]/dt1_h[idi], f, n, ii);
			if(Coltime_h[0] == 10.0) {
				if((accept == 0 || sgnt * dt1 < def_dtmin) && Ne1 >= 0){

					t1 += dt1;
					if(n >= 8) dt1 *= 0.55;
					if(n < 7) dt1 *= 1.3;
					if(sgnt * dt1 > sgnt * dt) dt1 = dt;
					if(sgnt * (t1 + dt1) > sgnt * dt) dt1 = dt - t1;
					if(sgnt * dt1 < def_dtmin) dt1 = sgnt * def_dtmin;

					xold_h[idi] = xt_h[idi];
					vold_h[idi] = vt_h[idi];
//if(idi == 12888 || idi == 11191) printf("update %d %d %.20g %.20g %.20g %.20g %.20g %.20g %g %g %g %d %d\n", idx, idi, xold_h[idi].x, xold_h[idi].y, xold_h[idi].z, vold_h[idi].x, vold_h[idi].y, vold_h[idi].z, t1 / dayUnit, dt1 / dayUnit, dt / dayUnit, f, n);
					dt1_h[idi] = dt1;
					t1_h[idi] = t1;
					Encpairs_h[idi + 6 * NT].y = -1;
				}
				else{
					if(n == 8 && Ne1 >= 0){
						dt1_h[idi] = 0.5 * dt1;
//if(id == 0) printf("continue %d %d %g %g %d %d\n", idx, idi, t1_h[idi] / dayUnit, dt1_h[idi] / dayUnit, f, n);
					}
				}
				if(sgnt * t1 >= sgnt * dt){
					//BS step finished
					x4_h[idi] = xt_h[idi];
					v4_h[idi] = vt_h[idi];
					Encpairs_h[idi + 7 * NT].y = -1;
//if(id == 0) printf("finished %d %d %g %.20g %.20g %.20g %.20g %.20g %.20g %d %d\n", idi, index_h[idi], x4_h[idi].w, x4_h[idi].x, x4_h[idi].y, x4_h[idi].z, v4_h[idi].x, v4_h[idi].y, v4_h[idi].z, f, n);
				}
				else{
//if(id == 0) printf("not finished %d %d %d %d %d\n", idx, idi, f, n, BSAstop_h[0]);
					if(BSAstop_h[0] != 3) BSAstop_h[0] = 0;
				}
			}
			else{
				dt1_h[idi] *= Coltime_h[0];
				BSAstop_h[0] = 2;
//if(id == 0) printf("reduce time step %d %g Coltime: %g %d %d noColl %d\n", idx, dt1_h[idi] / dayUnit,  Coltime_h[0], f, n, noColl);
			}
		}
	}
}

void BSA_setdt_cpu(double *dt1_h, double *t1_h, const double dt, const int N, double ksqMsun, double4 *x4_h, double4 *v4_h, double *dtgr_h, const int UseGR){


	#pragma omp parallel for
	for(int i = 0; i < N; ++i){
		if(UseGR == 1){// GR time rescale (Saha & Tremaine 1994)
			double c2 = def_cm * def_cm;
			double mu = ksqMsun;
			double rsq = x4_h[i].x * x4_h[i].x + x4_h[i].y * x4_h[i].y + x4_h[i].z * x4_h[i].z;
			double vsq = v4_h[i].x * v4_h[i].x + v4_h[i].y * v4_h[i].y + v4_h[i].z * v4_h[i].z;
			double ir = 1.0/sqrt(rsq);
			double ia = 2.0*ir-vsq/mu;
			dtgr_h[i] = 1.0 - 1.5 * mu * ia / c2;
		}
		else{
			dtgr_h[i] = 1.0;
		}
		dt1_h[i] = dt;
		t1_h[i] = 0.0;
	}
}

void Data::BSACall(const int st, const int b, const int Nm, const int si, const double t, const double FGt, int noColl){
	int N = N_h[0] + Nsmall_h[0];

	BSA_setdt_cpu (dt1_h, t1_h, dt_h[0] * FGt, N, def_ksq * Msun_h[0].x, xold_h, vold_h, dtgr_h, P.UseGR);
	BSAstop_h[0] = 0;
	for(int f = 0; f < 100000; ++f){
		for(int n = 1; n <= 8; ++n){
//printf("%d %d %d\n", f, n, BSAstop_h[0]);
			if(BSAstop_h[0] == 1) break; 
			if(BSAstop_h[0] == 3) break;
			if(Ncoll_m[0] > def_MaxColl) break;
			for(int idx = 0; idx < Nm; ++idx){ 
				BSAcc_cpu < 0 > (xold_h, vold_h, xold_h, vold_h, xp_h, vp_h, rcritv_h, Encpairs_h, Encpairs2_h, dt1_h, dtgr_h, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_h, Coltime_h, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl, idx);
			}
			for(int idx = 0; idx < Nm; ++idx){ 
				BSAcc_cpu < 1 > (xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcritv_h, Encpairs_h, Encpairs2_h, dt1_h, dtgr_h, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_h, Coltime_h, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl, idx);
			}
			for(int m = 2; m <= n; ++m){
				for(int idx = 0; idx < Nm; ++idx){ 
					BSAcc_cpu < 2 > (xold_h, vold_h, xt_h, vt_h, xp_h, vp_h, rcritv_h, Encpairs_h, Encpairs2_h, dt1_h, dtgr_h, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_h, Coltime_h, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl, idx);
				}
				for(int idx = 0; idx < Nm; ++idx){ 
					BSAcc_cpu < 2 > (xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, rcritv_h, Encpairs_h, Encpairs2_h, dt1_h, dtgr_h, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_h, Coltime_h, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl, idx);
				}
			}
			for(int idx = 0; idx < Nm; ++idx){ 
				BSAcc_cpu < 3 > (xold_h, vold_h, xt_h, vt_h, xp_h, vp_h, rcritv_h, Encpairs_h, Encpairs2_h, dt1_h, dtgr_h, Msun_h[0].x, st, N, NconstT, P.NencMax, BSAstop_h, Coltime_h, n, f, P.MinMass, P.UseTestParticles, dt_h[0], P.SLevels, noColl, idx);
			}
			for(int idx = 0; idx < Nm; ++idx){ 
				BSError_cpu (xold_h, vold_h, xp_h, vp_h, xt_h, vt_h, dx_h, dv_h, Encpairs_h, Encpairs2_h, st, N, n, f, idx);
			}
			for(int idx = 0; idx < Nm; ++idx){ 
				BSAccept_cpu (xt_h, vt_h, dx_h, dv_h, Encpairs_h, Encpairs2_h, dt1_h, st, N, n, idx);
			}
			for(int idx = 0; idx < Nm; ++idx){ 
				BSUpdate_cpu (random_h, xold_h, vold_h, x4_h, v4_h, xt_h, vt_h, rcrit_h, rcritv_h, index_h, spin_h, love_h, createFlag_h, Encpairs_h, Encpairs2_h, BSAstop_h, dt1_h, t1_h, dt_h[0] * FGt, Msun_h[0].x, U_h, st, N, NconstT, f, n, P.NencMax, Ncoll_m, Coll_h, t, aelimits_h, aecount_h, enccount_h, aecountT_h, enccountT_h, NWriteEnc_m, writeEnc_h, dtgr_h, Coltime_h, P.MinMass, P.UseTestParticles, P.SLevels, noColl, idx);
			}
//printf("A %d %d\n", BSAstop_h[0], Ncoll_m[0]);
			if(BSAstop_h[0] == 1) break;
			if(BSAstop_h[0] == 2) break;
			if(BSAstop_h[0] == 3) break;
			if(Ncoll_m[0] > def_MaxColl) break; 
		}
//printf("B %d %d\n", BSAstop_h[0], Ncoll_m[0]);
		if(BSAstop_h[0] == 1) break; 
		if(BSAstop_h[0] == 3) break;
		if(Ncoll_m[0] > def_MaxColl) break; 
	}
}

