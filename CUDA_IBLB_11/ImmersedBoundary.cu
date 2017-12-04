#include <cmath>
#include <cstdlib>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImmersedBoundary.cuh"

#define PI 3.14159
//__device__ const double RHO_0 = 1.;
//__device__ const double C_S = 0.57735;

__device__ const double c_l[9 * 2] =		//VELOCITY COMPONENTS
{
	0.,0. ,
	1.,0. , 0.,1. , -1.,0. , 0.,-1. ,
	1.,1. , -1.,1. , -1.,-1. , 1.,-1.
};

__device__ double delta(const double & xs, const double & ys, const int & x, const int & y)
{
	double deltax(0.), deltay(0.), delta(0.);

	double dx = abs(x - xs);
	double dy = abs(y - ys);

	if (dx <= 1.5)
	{
		if (dx <= 0.5)
		{
			deltax = (1. / 3.)*(1. + sqrt(-3. * dx*dx + 1.));
		}
		else deltax = (1. / 6.)*(5. - 3. * dx - sqrt(-3. * (1. - dx)*(1. - dx) + 1.));
	}

	if (dy <= 1.5)
	{
		if (dy <= 0.5)
		{
			deltay = (1. / 3.)*(1. + sqrt(-3. * dy*dy + 1.));
		}
		else deltay = (1. / 6.)*(5. - 3. * dy - sqrt(-3. * (1. - dy)*(1. - dy) + 1.));
	}

	delta = deltax * deltay;

	return delta;
}

__global__ void interpolate(const double * rho, const double * u, const int * Ns, const double * u_s, double * F_s, const double * s, const int * XDIM)
{

	int i(0), j(0), k(0), x0(0), y0(0), x(0), y(0);

	double xs(0.), ys(0.);


	k = blockIdx.x*blockDim.x + threadIdx.x;


	{
		F_s[2 * k + 0] = 0.;
		F_s[2 * k + 1] = 0.;

		xs = s[k * 2 + 0];
		ys = s[k * 2 + 1];

		x0 = nearbyint(xs);
		y0 = nearbyint(ys);

		for (i = 0; i < 9; i++)
		{
			x = nearbyint(x0 + c_l[i * 2 + 0]);
			y = nearbyint(y0 + c_l[i * 2 + 1]);

			j = y*XDIM[0] + x;

			//std::cout << delta << std::endl;

			F_s[2 * k + 0] += 2.*(1. * 1. * delta(xs, ys, x, y))*rho[j] * (u_s[2 * k + 0] - u[2 * j + 0]);
			F_s[2 * k + 1] += 2.*(1. * 1. * delta(xs, ys, x, y))*rho[j] * (u_s[2 * k + 1] - u[2 * j + 1]);
		}

	}
}

__global__ void spread(const double * rho, double * u, const double * f, const int * Ns, const double * u_s, const double * F_s, double * force, const double * s, const int * XDIM, double * Q)
{
	int i(0), j(0), k(0), x(0), y(0);

	double xs(0.), ys(0.);

	double momentum[2] = { 0,0 };

	j = blockIdx.x*blockDim.x + threadIdx.x;


	{
		force[2 * j + 0] = 0.;
		force[2 * j + 1] = 0.;


		x = j%XDIM[0];
		y = (j - j%XDIM[0]) / XDIM[0];

		for (k = 0; k < Ns[0]; k++)
		{
			xs = s[k * 2 + 0];
			ys = s[k * 2 + 1];

			force[2 * j + 0] += F_s[2 * k + 0] * delta(xs, ys, x, y)*1.;
			force[2 * j + 1] += F_s[2 * k + 1] * delta(xs, ys, x, y)*1.;
		}

		momentum[0] = 0.;
		momentum[1] = 0.;

		for (i = 0; i < 9; i++)
		{
			momentum[0] += c_l[i * 2 + 0] * f[9 * j + i];
			momentum[1] += c_l[i * 2 + 1] * f[9 * j + i];
		}

		u[2 * j + 0] = (momentum[0] + 0.5*force[2 * j + 0]) / rho[j];
		u[2 * j + 1] = (momentum[1] + 0.5*force[2 * j + 1]) / rho[j];

		if (x == XDIM[0] - 5)
		{

				Q[0] += u[2 * j + 0];

		}
	}

}


