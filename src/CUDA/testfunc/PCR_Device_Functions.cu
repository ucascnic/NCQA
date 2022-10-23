#include <cstdio>
#include "PCR_Device_functions.h"

__global__ void list_print(int nmax, float * in) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    printf("Thread %i shows %f \n", i, in[i]);
}


__global__ void Solve_Kernel(
    float * alist, float * blist, float * clist, float * dlist, float * xlist,
    int iter_max, int DMax) {

    int idx_row = blockIdx.x*blockDim.x + threadIdx.x;
    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    float a1, b1, c1, d1;
    float k01, k21, c01, a21, d01, d21;

    bool next_or_ot = true;
    int accum;

    for (int iter = 0; iter < iter_max; iter++) {

        if ( next_or_ot ) {

            next_stride = stride<<1;

            // 1    for updating 'a'
            if ((idx_row - stride)<0) {
            // 1.1  if it is the 'first' line
                a1 = 0.0f;
                k01 = 0.0f;
                c01 = 0.0f;
                d01 = 0.0f;
            } else if ((idx_row - next_stride)<0) {
            // 1.2  if no place for 'a'
                a1 = 0.0f;
                k01 = alist[idx_row]/blist[idx_row - stride];
                c01 = clist[idx_row - stride]*k01;
                d01 = dlist[idx_row - stride]*k01;
            } else {
            // 1.3  for rest general rows
                k01 = alist[idx_row]/blist[idx_row - stride];
                a1 = -alist[idx_row - stride]*k01;
                c01 = clist[idx_row - stride]*k01;
                d01 = dlist[idx_row - stride]*k01;
            }

            // 2    for updating 'c'
            if ((idx_row + stride)>row_max) {
            // 2.1  if it is the 'last' line
                c1 = 0.0f;
                k21 = 0.0f;
                a21 = 0.0f;
                d21 = 0.0f;
            } else if ((idx_row + next_stride)>row_max) {
                c1 = 0.0f;
                k21 = clist[idx_row]/blist[idx_row + stride];
                a21 = alist[idx_row + stride]*k21;
                d21 = dlist[idx_row + stride]*k21;
            } else {
                k21 = clist[idx_row]/blist[idx_row + stride];
                c1 = -clist[idx_row + stride]*k21;
                a21 = alist[idx_row + stride]*k21;
                d21 = dlist[idx_row + stride]*k21;
            }
            // 3   for updating 'b'
            b1 = blist[idx_row] - c01 - a21;
            // 4   for updating 'd'
            d1 = dlist[idx_row] - d01 - d21;

            stride = next_stride;

            //Determine if this line has reached the bi-set
            int pos = idx_row-2*stride;
            accum = 0;
            for ( size_t iter = 0; iter<5; iter++ ) {
                if (pos >=0 && pos < DMax) accum++;
                pos+=stride;
            }
            if (accum < 3) {
                next_or_ot = false;//Turn of for ever
            }

        }

        __syncthreads();
        __threadfence();
        alist[idx_row] = a1;
        blist[idx_row] = b1;
        clist[idx_row] = c1;
        dlist[idx_row] = d1;

    }

    if ( accum==1 ) {
        xlist[idx_row] = dlist[idx_row] / blist[idx_row];
    } else if ( (idx_row-stride)<0 ) {
        int i = idx_row; int k = idx_row+stride;
        float f = clist[i]/blist[k];
        xlist[i] = (dlist[i]-dlist[k]*f)/(blist[i]-alist[k]*f);
    } else {
        int i = idx_row - stride; int k = idx_row;
        float f = alist[k]/blist[i];
        xlist[k] = (dlist[k]-dlist[i]*f)/(blist[k]-clist[i]*f);
    }

}


__global__ void Solve_KernelD(
    double * alist, double * blist, double * clist, double * dlist, double * xlist,
    int iter_max, int DMax) {

    int idx_row = blockIdx.x*blockDim.x + threadIdx.x;

    if (idx_row  >= DMax)
        return ;

    int row_max = DMax - 1;

    int stride = 1;
    int next_stride = stride;

    double a1, b1, c1, d1;
    double k01, k21, c01, a21, d01, d21;

    bool next_or_ot = true;
    int accum;

    for (int iter = 0; iter < iter_max; iter++) {

        if ( next_or_ot ) {

            next_stride = stride<<1;

            // 1    for updating 'a'
            if ((idx_row - stride)<0) {
            // 1.1  if it is the 'first' line
                a1 = 0.0;
                k01 = 0.0;
                c01 = 0.0;
                d01 = 0.0;
            } else if ((idx_row - next_stride)<0) {
            // 1.2  if no place for 'a'
                a1 = 0.0;
                k01 = alist[idx_row]/blist[idx_row - stride];
                c01 = clist[idx_row - stride]*k01;
                d01 = dlist[idx_row - stride]*k01;
            } else {
            // 1.3  for rest general rows
                k01 = alist[idx_row]/blist[idx_row - stride];
                a1 = -alist[idx_row - stride]*k01;
                c01 = clist[idx_row - stride]*k01;
                d01 = dlist[idx_row - stride]*k01;
            }

            // 2    for updating 'c'
            if ((idx_row + stride)>row_max) {
            // 2.1  if it is the 'last' line
                c1 = 0.0;
                k21 = 0.0;
                a21 = 0.0;
                d21 = 0.0;
            } else if ((idx_row + next_stride)>row_max) {
                c1 = 0.0;
                k21 = clist[idx_row]/blist[idx_row + stride];
                a21 = alist[idx_row + stride]*k21;
                d21 = dlist[idx_row + stride]*k21;
            } else {
                k21 = clist[idx_row]/blist[idx_row + stride];
                c1 = -clist[idx_row + stride]*k21;
                a21 = alist[idx_row + stride]*k21;
                d21 = dlist[idx_row + stride]*k21;
            }
            // 3   for updating 'b'
            b1 = blist[idx_row] - c01 - a21;
            // 4   for updating 'd'
            d1 = dlist[idx_row] - d01 - d21;

            stride = next_stride;

            //Determine if this line has reached the bi-set
            int pos = idx_row-2*stride;
            accum = 0;
            for ( size_t iter = 0; iter<5; iter++ ) {
                if (pos >=0 && pos < DMax) accum++;
                pos+=stride;
            }
            if (accum < 3) {
                next_or_ot = false;//Turn of for ever
            }

        }

        __syncthreads();__threadfence();

        alist[idx_row] = a1;
        blist[idx_row] = b1;
        clist[idx_row] = c1;
        dlist[idx_row] = d1;

    }

    if ( accum==1 ) {
        xlist[idx_row] = dlist[idx_row] / blist[idx_row];
    } else if ( (idx_row-stride)<0 ) {
        int i = idx_row; int k = idx_row+stride;
        double f = clist[i]/blist[k];
        xlist[i] = (dlist[i]-dlist[k]*f)/(blist[i]-alist[k]*f);
    } else {
        int i = idx_row - stride; int k = idx_row;
        double f = alist[k]/blist[i];
        xlist[k] = (dlist[k]-dlist[i]*f)/(blist[k]-clist[i]*f);
    }

}

