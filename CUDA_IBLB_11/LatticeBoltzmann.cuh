#ifndef LATTICEBOLTZMANN_CUH
#define LATTICEBOLTZMANN_CUH

__global__ void equilibrium(const double * u, const double * rho, double * f0, const double * force, double * F, int * XDIM, int * YDIM, double * TAU);

__global__ void collision(const double * f0, const double * f, double * f1, const double * F, double * TAU, double * TAU2, int * XDIM, int * YDIM, int * it);

__global__ void streaming(const double * f1, double * f, int * XDIM, int * YDIM);

__global__ void macro(const double * f, double * u, double * rho, int * XDIM, int * YDIM);

#endif // !LATTICEBOLTZMANN_H
