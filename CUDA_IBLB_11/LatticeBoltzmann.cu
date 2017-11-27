#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "LatticeBoltzmann.cuh"



__device__ const double C_S = 0.57735;
//__device__ const double TAU2 = 0.505556;
//__device__ const double RHO_0 = 1.;

__device__ const double c_l[9 * 2] =		//VELOCITY COMPONENTS
{
	0.,0. ,
	1.,0. , 0.,1. , -1.,0. , 0.,-1. ,
	1.,1. , -1.,1. , -1.,-1. , 1.,-1.
};

__device__ const double t[9] =					//WEIGHT VALUES
{
	4. / 9,
	1. / 9, 1. / 9, 1. / 9, 1. / 9,
	1. / 36, 1. / 36, 1. / 36, 1. / 36
};


__global__ void equilibrium(const double * u, const double * rho, double * f0, const double * force, double * F, int * XDIM, int * YDIM, double * TAU)
{
	unsigned int i(0), j(0);

	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	double vec[2] = { 0,0 };

	
	{
		j = threadnum;

		for (i = 0; i < 9; i++)
		{
			
			f0[9 * j + i] = rho[j] * t[i] * (1
				+ (u[2 * j + 0] * c_l[2 * i + 0] + u[2 * j + 1] * c_l[2 * i + 1]) / (C_S*C_S)
				+ (u[2 * j + 0] * c_l[2 * i + 0] + u[2 * j + 1] * c_l[2 * i + 1])*(u[2 * j + 0] * c_l[2 * i + 0] + u[2 * j + 1] * c_l[2 * i + 1]) / (2 * C_S*C_S*C_S*C_S)
				- (u[2 * j + 0] * u[2 * j + 0] + u[2 * j + 1] * u[2 * j + 1]) / (2 * C_S*C_S));
			

			vec[0] = (c_l[i * 2 + 0] - u[i * 2 + 0]) / (C_S*C_S) + (c_l[i * 2 + 0] * u[i * 2 + 0] + c_l[i * 2 + 1] * u[i * 2 + 1]) / (C_S*C_S*C_S*C_S) * c_l[i * 2 + 0];
			vec[1] = (c_l[i * 2 + 1] - u[i * 2 + 1]) / (C_S*C_S) + (c_l[i * 2 + 0] * u[i * 2 + 0] + c_l[i * 2 + 1] * u[i * 2 + 1]) / (C_S*C_S*C_S*C_S) * c_l[i * 2 + 1];

			F[9 * j + i] = (1. - 1. / (2. * TAU[0]))*t[i] * (vec[0] * force[j * 2 + 0] + vec[1] * force[j * 2 + 1]);
			
		}
	}
}

__global__ void collision(const double * f0, const double * f, double * f1, const double * F, double * TAU, double * TAU2, int * XDIM, int * YDIM, int * it)
{
	unsigned int j(0);

	//double rho_set = 1.;
	//double u_set[2] = { 0.00004,0. };
	//double u_s[2] = { 0.,0. };

	double omega_plus = 1 / TAU[0];
	double omega_minus = 1 / TAU2[0];

	double f_plus(0.), f_minus(0.), f0_plus(0.), f0_minus(0.);

	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	{
		j = threadnum;

		//for (i = 0; i < 9; i++)
		{
			//f1[9 * j + i] = (1 - (1 / TAU[0]))*f[9 * j + i] + (1 / TAU[0])*f0[9 * j + i] + F[j * 9 + i];

			f1[9 * j + 0] = f[9 * j + 0] - omega_plus*(f[9 * j + 0] - f0[9 * j + 0]) + F[j * 9 + 0];

			f_plus = (f[9 * j + 1] + f[9 * j + 3]) / 2.;
			f_minus = (f[9 * j + 1] - f[9 * j + 3]) / 2.;
			f0_plus = (f0[9 * j + 1] + f0[9 * j + 3]) / 2.;
			f0_minus = (f0[9 * j + 1] - f0[9 * j + 3]) / 2.;

			f1[9 * j + 1] = f[9 * j + 1] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 3] = f[9 * j + 3] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

			f_plus = (f[9 * j + 2] + f[9 * j + 4]) / 2.;
			f_minus = (f[9 * j + 2] - f[9 * j + 4]) / 2.;
			f0_plus = (f0[9 * j + 2] + f0[9 * j + 4]) / 2.;
			f0_minus = (f0[9 * j + 2] - f0[9 * j + 4]) / 2.;

			f1[9 * j + 2] = f[9 * j + 2] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 4] = f[9 * j + 4] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

			f_plus = (f[9 * j + 5] + f[9 * j + 7]) / 2.;
			f_minus = (f[9 * j + 5] - f[9 * j + 7]) / 2.;
			f0_plus = (f0[9 * j + 5] + f0[9 * j + 7]) / 2.;
			f0_minus = (f0[9 * j + 5] - f0[9 * j + 7]) / 2.;

			f1[9 * j + 5] = f[9 * j + 5] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 7] = f[9 * j + 7] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

			f_plus = (f[9 * j + 6] + f[9 * j + 8]) / 2.;
			f_minus = (f[9 * j + 6] - f[9 * j + 8]) / 2.;
			f0_plus = (f0[9 * j + 6] + f0[9 * j + 8]) / 2.;
			f0_minus = (f0[9 * j + 6] - f0[9 * j + 8]) / 2.;

			f1[9 * j + 6] = f[9 * j + 6] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

			f_minus *= -1.;
			f0_minus *= -1.;

			f1[9 * j + 8] = f[9 * j + 8] - omega_plus*(f_plus - f0_plus) - omega_minus*(f_minus - f0_minus);

		}

		//--------------------------------ZOU-HE VELOCITY BOUNDARY-------------------------
		/*
		if (j%XDIM[0] == 0)										//LEFT
		{
		
		//rho_set = 1 / (1 - u_set[0])*(f[9 * j + 0] + f[9 * j + 2] + f[9 * j + 4] + 2 * (f[9 * j + 3] + f[9 * j + 6] + f[9 * j + 7]));
		rho_set = RHO_0;
		f1[9 * j + 1] = f[9 * j + 3] + (2./3.)*rho_set*u_set[0];

		f1[9 * j + 5] = f[9 * j + 7] - 0.5*(f[9 * j + 2] - f[9 * j + 4]) + 0.5*rho_set*u_set[1] + (1. / 6.)*rho_set*u_set[0];

		f1[9 * j + 8] = f[9 * j + 6] + 0.5*(f[9 * j + 2] - f[9 * j + 4]) - 0.5*rho_set*u_set[1] + (1. / 6.)*rho_set*u_set[0];
		}
		*/
		/*
		if (j % XDIM[0] == XDIM[0]-1 )										//RIGHT
		{
		rho_set = RHO_0;

		u_s[0] = 1. - (f[9 * j + 0] + f[9 * j + 2] + f[9 * j + 4] + 2. * (f[9 * j + 1] + f[9 * j + 5] + f[9 * j + 8]))/rho_set;

		u_s[1] = 0.;

		f1[9 * j + 3] = f[9 * j + 1] + (2. / 3.)*rho_set*u_s[0];

		f1[9 * j + 7] = f[9 * j + 5] - 0.5*(f[9 * j + 4] - f[9 * j + 2]) + 0.5*rho_set*u_s[1] + (1. / 6.)*rho_set*u_s[0];

		f1[9 * j + 6] = f[9 * j + 8] + 0.5*(f[9 * j + 4] - f[9 * j + 2]) - 0.5*rho_set*u_s[1] + (1. / 6.)*rho_set*u_s[0];
		}
		*/
	}
}

__global__ void streaming(const double * f1, double * f, int * XDIM, int * YDIM)
{
	
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0), k(0);
	unsigned int jstream(0);
	bool back(0), thru(0), done(0), slip(0);
	bool up(0), down(0), left(0), right(0);

	int x(0), y(0);

	
	{
		j = threadnum;

		x = j%XDIM[0];
		y = (j - j%XDIM[0]) / XDIM[0];

		//------------------------------------WALL CONDITIONS------------------------------------------------

		up = 0;
		down = 0;
		left = 0;
		right = 0;

		if (y == YDIM[0] - 1) up = 1;
		if (y == 0) down = 1;
		if (x == 0) left = 1;
		if (x == XDIM[0] - 1) right = 1;

		for (i = 0; i < 9; i++)
		{
			//cout << i << endl;

			back = 0;
			thru = 0;
			done = 0;
			slip = 0;

			//---------------------------------------------------MID GRID NON-SLIP BOUNDARY------------------------------

			if (down || up || left || right)
			{
				switch (i)
				{
				case 0: break;

				case 1:
					if (right) { thru = 1; break; }
					break;
				case 2:
					if (up) { slip = 1; break; }
					break;
				case 3:
					if (left) { thru = 1; break; }
					break;
				case 4:
					if (down) { back = 1; break; }
					break;
				case 5:
					/*
					if (up && left)
					{
						jstream = j - (XDIM[0] - 1)*c_l[7 * 2 + 0] + XDIM[0] * c_l[7 * 2 + 1]; //THROUGH STREAM 7
						k = 7;
						done = 1;
						break;
					}
					*/
					if (up)
					{
						slip = 1;
						break;
					}
					else if (right)
					{
						thru = 1;
						break;
					}

					break;
				case 6:
					/*
					if (up && right)
					{
						jstream = j - (XDIM[0] - 1)*c_l[8 * 2 + 0] + XDIM[0] * c_l[8 * 2 + 1]; //THROUGH STREAM 8
						k = 8;
						done = 1;
						break;
					}
					*/
					if (up)
					{
						slip = 1;
						break;
					}
					else if (left)
					{
						thru = 1;
						break;
					}

					break;
				case 7:
					/*
					if (down && right)
					{
						jstream = j - (XDIM[0] - 1)*c_l[5 * 2 + 0] + XDIM[0] * c_l[5 * 2 + 1]; //THROUGH STREAM 5
						k = 5;
						done = 1;
						break;
					}
					*/
					if (down)
					{
						back = 1;
						break;
					}
					else if (left)
					{
						thru = 1;
						break;
					}
					
					break;
				case 8:
					/*
					if (down && left)
					{
						jstream = j - (XDIM[0] - 1)*c_l[6 * 2 + 0] + XDIM[0] * c_l[6 * 2 + 1]; //THROUGH STREAM 6
						k = 6;
						done = 1;
						break;
					}
					*/
					if (down)
					{
						back = 1;
						break;
					}
					else if (right)
					{
						thru = 1;
						break;
					}
					
					break;
				}

			}

			
			//--------------------------------------------------STREAMING CALCULATIONS-------------------------------

			if (back && !done)
			{
				jstream = j; //BACK STREAM

				if (i == 1) k = 3;
				if (i == 2) k = 4;
				if (i == 3) k = 1;
				if (i == 4) k = 2;
				if (i == 5) k = 7;
				if (i == 6) k = 8;
				if (i == 7) k = 5;
				if (i == 8) k = 6;
			}
			else if (slip && !done)
			{
				jstream = j; //SLIP STREAM

				if (i == 1) k = 3;
				if (i == 2) k = 4;
				if (i == 3) k = 1;
				if (i == 4) k = 2;
				if (i == 5) k = 8;
				if (i == 6) k = 7;
				if (i == 7) k = 6;
				if (i == 8) k = 5;
			}
			else if (thru && !done)
			{
				jstream = j - (XDIM[0]-1)*c_l[i * 2 + 0] + XDIM[0]*c_l[i * 2 + 1]; //THROUGH STREAM

				k = i;
			}
			else if (!done)
			{
				jstream = j + c_l[i * 2 + 0] + XDIM[0]*c_l[i * 2 + 1]; //NORMAL STREAM

				k = i;
			}

			f[9 * jstream + k] = f1[9 * j + i];								//STREAM TO ADJACENT CELL IN DIRECTION OF MOVEMENT
		}
	}

}

__global__ void macro(const double * f, double * u, double * rho, int * XDIM, int * YDIM)
{
	int threadnum = blockIdx.x*blockDim.x + threadIdx.x;

	unsigned int i(0), j(0);

	double momentum[2] = { 0,0 };

	{
		j = threadnum;

		rho[j] = 0;

		u[2 * j + 0] = 0;
		u[2 * j + 1] = 0;

		momentum[0] = 0;
		momentum[1] = 0;

		for (i = 0; i < 9; i++)
		{
			rho[j] += f[9 * j + i];

			momentum[0] += c_l[i * 2 + 0] * f[9 * j + i];
			momentum[1] += c_l[i * 2 + 1] * f[9 * j + i];
		}

		u[2 * j + 0] = momentum[0] / rho[j];
		u[2 * j + 1] = momentum[1] / rho[j];

		
	}
}