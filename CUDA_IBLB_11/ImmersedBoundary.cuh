#ifndef IMMERSEDBOUNDARY_CUH
#define IMMERSEDBOUNDARY_CUH

__device__ double delta(const double & xs, const double & ys, const int & x, const int & y);

__global__ void interpolate(const double * rho, const double * u, const int * Ns, const double * b_points, double * F_s, const int * XDIM);

__global__ void spread(const double * rho, double * u, const double * f, const int * Ns, const double * F_s, double * force, const double * b_points, const int * XDIM, double * Q);


#endif // !IMMERSEDBOUNDARY_H