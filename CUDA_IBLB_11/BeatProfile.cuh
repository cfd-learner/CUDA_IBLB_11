#ifndef BEATPROFILE_CUH
#define BEATPROFILE_CUH

__global__ void define_filament(const int * m, const int * it, const double * offset, double * s, double * lasts);

__global__ void define_boundary(const int * m, const double * boundary, const int * c_num, double * b_points);


#endif // !BEATPROFILE_H