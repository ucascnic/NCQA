

#ifndef PCR_Device_Functions_cuh
#define PCR_Device_Functions_cuh

__global__ void list_print(int nmax, float *in);

__global__ void Solve_Kernel(float * alist, float * blist, float * clist, float * dlist, float * xlist, int iter_max, int DMax);

__global__ void Solve_KernelD(double * alist, double * blist, double * clist,
                              double * dlist, double * xlist, int iter_max, int DMax);

#endif

