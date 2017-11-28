#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "seconds.h"

#define Ns 10000
#define Nb 100
#define PI 3.142
#define LENGTH 115
#define T 100000


__global__ void define_filament(const int * m, const int * it, const double * offset, double * s, double * lasts)
{
	int n(0);

	double arcl(0.);

	double A_mn[7 * 2 * 3] = 
	{
		-0.449,	 0.130, -0.169,	 0.063, -0.050, -0.040, -0.068,
		 2.076, -0.003,	 0.054,	 0.007,	 0.026,	 0.022,	 0.010,
		-0.072, -1.502,	 0.260, -0.123,	 0.011, -0.009,	 0.196,
		-1.074, -0.230, -0.305, -0.180, -0.069,	 0.001, -0.080,
		 0.658,	 0.793, -0.251,	 0.049,	 0.009,	 0.023, -0.111,
		 0.381,	 0.331,	 0.193,	 0.082,	 0.029,	 0.002,	 0.048,
	};

	double B_mn[7 * 2 * 3] =
	{
		0.0,	-0.030, -0.093,	 0.037,	 0.062,	 0.016, -0.065,
		0.0,	 0.080, -0.044, -0.017,	 0.052,	 0.007,	 0.051,
		0.0,	 1.285,	-0.036, -0.244, -0.093, -0.137,	 0.095,
		0.0,	-0.298,	 0.513,	 0.004, -0.222,	 0.035, -0.128,
		0.0,	-1.034,	 0.050,	 0.143,	 0.043,	 0.098, -0.054,
		0.0,	 0.210, -0.367,	 0.009,	 0.120, -0.024,	 0.102,
	};

	double a_n[2 * 7];
	double b_n[2 * 7];

	int threadnum = blockDim.x*blockIdx.x + threadIdx.x;

	int k = threadnum;

	{
		arcl = 1.*k / Ns;

		for (n = 0; n < 7; n++)
		{
			a_n[2 * n + 0] = 0.;
			b_n[2 * n + 0] = 0.;

			a_n[2 * n + 0] += A_mn[n + 14 * 0 + 7 * 0] * pow(arcl, 0 + 1);
			b_n[2 * n + 0] += B_mn[n + 14 * 0 + 7 * 0] * pow(arcl, 0 + 1);

			a_n[2 * n + 0] += A_mn[n + 14 * 1 + 7 * 0] * pow(arcl, 1 + 1);
			b_n[2 * n + 0] += B_mn[n + 14 * 1 + 7 * 0] * pow(arcl, 1 + 1);

			a_n[2 * n + 0] += A_mn[n + 14 * 2 + 7 * 0] * pow(arcl, 2 + 1);
			b_n[2 * n + 0] += B_mn[n + 14 * 2 + 7 * 0] * pow(arcl, 2 + 1);

			a_n[2 * n + 1] = 0.;
			b_n[2 * n + 1] = 0.;

			a_n[2 * n + 1] += A_mn[n + 14 * 0 + 7 * 1] * pow(arcl, 0 + 1);
			b_n[2 * n + 1] += B_mn[n + 14 * 0 + 7 * 1] * pow(arcl, 0 + 1);

			a_n[2 * n + 1] += A_mn[n + 14 * 1 + 7 * 1] * pow(arcl, 1 + 1);
			b_n[2 * n + 1] += B_mn[n + 14 * 1 + 7 * 1] * pow(arcl, 1 + 1);

			a_n[2 * n + 1] += A_mn[n + 14 * 2 + 7 * 1] * pow(arcl, 2 + 1);
			b_n[2 * n + 1] += B_mn[n + 14 * 2 + 7 * 1] * pow(arcl, 2 + 1);

		}

		s[5 * (k + m[0] * Ns) + 0] = 1.*LENGTH*a_n[2 * 0 + 0] * 0.5 + offset[0];
		s[5 * (k + m[0] * Ns) + 1] = 1.*LENGTH*a_n[2 * 0 + 1] * 0.5 + 2;
		s[5 * (k + m[0] * Ns) + 2] = LENGTH*arcl;

		for (n = 1; n < 7; n++)
		{
			s[5 * (k + m[0] * Ns) + 0] += 1.*LENGTH*(a_n[2 * n + 0] * cos(n*2.*PI*it[0] / T) + b_n[2 * n + 0] * sin(n*2.*PI*it[0] / T));
			s[5 * (k + m[0] * Ns) + 1] += 1.*LENGTH*(a_n[2 * n + 1] * cos(n*2.*PI*it[0] / T) + b_n[2 * n + 1] * sin(n*2.*PI*it[0] / T));
		}

		if (it > 0)
		{
			s[5 * (k + m[0] * Ns) + 3] = s[5 * (k + m[0] * Ns) + 0] - lasts[2 * (k + m[0] * Ns) + 0];
			s[5 * (k + m[0] * Ns) + 4] = s[5 * (k + m[0] * Ns) + 1] - lasts[2 * (k + m[0] * Ns) + 1];
		}

		lasts[2 * (k + m[0] * Ns) + 0] = s[5 * (k + m[0] * Ns) + 0];
		lasts[2 * (k + m[0] * Ns) + 1] = s[5 * (k + m[0] * Ns) + 1];
	}
}

__global__ void define_boundary(const int * m, const double * boundary, const int * c_num, double * b_points)
{
	int j(0), k(0);
	double b_length(0.);
	double step(1.);

	int threadnum = blockDim.x*blockIdx.x + threadIdx.x;

	k = threadnum;

	//for (k = 1; k < Nb; k++)
	if (k == 0)
	{
		b_points[4 * (k + m[0] * Nb) + 0] = boundary[5 * (1 + m[0] * Ns) + 0];
		b_points[4 * (k + m[0] * Nb) + 1] = boundary[5 * (1 + m[0] * Ns) + 1];

		b_points[4 * (k + m[0] * Nb) + 2] = boundary[5 * (1 + m[0] * Ns) + 3];
		b_points[4 * (k + m[0] * Nb) + 3] = boundary[5 * (1 + m[0] * Ns) + 4];
	}
	else
	{
		b_length = k*step;

		for (j = (1 + m[0] * Ns); j < c_num[0]*Ns; j++)
		{
			if (abs(boundary[5 * j + 2] - b_length) < 0.01)
			{
				b_points[4 * (k + m[0] * Nb) + 0] = boundary[5 * j + 0];
				b_points[4 * (k + m[0] * Nb) + 1] = boundary[5 * j + 1];

				b_points[4 * (k + m[0] * Nb) + 2] = boundary[5 * j + 3];
				b_points[4 * (k + m[0] * Nb) + 3] = boundary[5 * j + 4];

				j = c_num[0]*Ns;
			}
			else
			{
				b_points[4 * (k + m[0] * Nb) + 0] = 0.;
				b_points[4 * (k + m[0] * Nb) + 1] = 150.;

				b_points[4 * (k + m[0] * Nb) + 2] = 0.1;
				b_points[4 * (k + m[0] * Nb) + 3] = 0.1;
			}
		}
	}
}

