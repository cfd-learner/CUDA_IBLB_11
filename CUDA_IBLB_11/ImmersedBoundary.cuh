#ifndef IMMERSEDBOUNDARY_CUH
#define IMMERSEDBOUNDARY_CUH

__device__ double delta(const double & xs, const double & ys, const int & x, const int & y);

__global__ void interpolate(const double * rho, const double * u, const int * Ns, const double * u_s, double * F_s, const double * s, const int * XDIM);

__global__ void spread(const double * rho, double * u, const double * f, const int * Ns, const double * u_s, const double * F_s, double * force, const double * s, const int * XDIM, double * Q);


#endif // !IMMERSEDBOUNDARY_H