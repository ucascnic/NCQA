#ifndef BACKUP_H
#define BACKUP_H
#include "cu_cr_solver.h"
#include "cu_cr_internal.h"
#include "cu_cr_functors.cu"

#include <cuda.h>
#include <math.h>
#include <thread>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>

/*
* For method documentation see cu_cr_internal.h unless otherwise specified.
*/
__global__
void
point_dvd(double *d_vect_d,
                         double *d_vect_b,
                         double *d_vect_results,
                         int size){

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n>=size)
        return;
    if (d_vect_b[n] == 0.0)
    {
        d_vect_results[n] = 0.0;
        return;
    }
    d_vect_results[n] = d_vect_d[n]/d_vect_b[n];
}

__global__
void
point_alpha(double *d_vect_d,
                         double *d_vect_b,
                         double *d_vect_results,
                         int size){

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n>=size)
        return;
    if (d_vect_b[n] == 0.0)
    {
        d_vect_results[n] = 0.0;
        return;
    }
    d_vect_results[n] = -d_vect_d[n]/d_vect_b[n];
}

__global__
void
point_mul_plus(double *d_vect_d,
               double *d_vect_b,
               double *d_vect_results,
               double *res,
               int size){

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n>=size)
        return;

    d_vect_results[n] =  d_vect_d[n]*d_vect_b[n];
    res[n] += d_vect_results[n];
}

__global__
void
point_mul(double *d_vect_d,
               double *d_vect_b,
               double *d_vect_results,
               int size){

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n>=size)
        return;

    d_vect_results[n] =  d_vect_d[n]*d_vect_b[n];
}
namespace cyclic_reduction{

void Solve2(int size, double *d_vect_a,
               double * d_vect_b,
               double * d_vect_c,
               double * d_vect_d,
               double * d_vect_results,
           thrust::device_vector<double>& d_vect_x,
            thrust::device_vector<double>& d_vect_a_prime,
            thrust::device_vector<double>& d_vect_c_prime,
            thrust::device_vector<double>& d_vect_temp){




//Define and create Cuda Streams

    cudaStream_t s1,s2,s3;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);

//Foward Reduction Phase

    dim3 block((size)/128+ 1);
    int level = 1;
    while(level < size){

    //AlphaBeta Methods

        cudaMemcpy(thrust::raw_pointer_cast(d_vect_x.data()),d_vect_d,
                   sizeof(double)*size,cudaMemcpyDeviceToDevice);

        LowerAlphaBeta(size,level,
             thrust::device_pointer_cast(d_vect_a),
             d_vect_a_prime.data(),
             thrust::device_pointer_cast(d_vect_b)
        );

        UpperAlphaBeta(size, level,
             thrust::device_pointer_cast(d_vect_b),
             thrust::device_pointer_cast(d_vect_c),
            d_vect_c_prime.data()
        );


    //Front Methods

        MainFront(size, level,
            d_vect_a_prime.data(),
             thrust::device_pointer_cast(d_vect_b),
             thrust::device_pointer_cast(d_vect_c),
            d_vect_temp.data()
        );

        SolutionFront(size, level,
            d_vect_a_prime.data(),
            thrust::device_pointer_cast(d_vect_d),
            d_vect_x.data(),
            d_vect_temp.data()
        );

        LowerFront(size, level,
             thrust::device_pointer_cast(d_vect_a),
            d_vect_a_prime.data()
        );

    //Back Methods

        MainBack(size, level,
             thrust::device_pointer_cast(d_vect_a),
            d_vect_c_prime.data(),
             thrust::device_pointer_cast(d_vect_b),
            d_vect_temp.data()
        );

        SolutionBack(size, level,
            d_vect_c_prime.data(),
            thrust::device_pointer_cast(d_vect_d),
            d_vect_x.data(),
            d_vect_temp.data()
        );

        UpperBack(size, level,
             thrust::device_pointer_cast(d_vect_c),
             d_vect_c_prime.data()
        );


        cudaMemcpy(d_vect_a,thrust::raw_pointer_cast(d_vect_a_prime.data()),
                   sizeof(double)*size,cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vect_c,thrust::raw_pointer_cast(d_vect_c_prime.data()),
                   sizeof(double)*size,cudaMemcpyDeviceToDevice);

        cudaMemcpy(d_vect_d,thrust::raw_pointer_cast(d_vect_x.data()),
                   sizeof(double)*size,cudaMemcpyDeviceToDevice);
        level *= 2;
    }



    point_dvd<<<block,128>>>(d_vect_d,d_vect_b,d_vect_results,size);



    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);

    return ;
}
void Solve(int size, thrust::device_vector<double> &d_vect_a,
               thrust::device_vector<double> & d_vect_b,
               thrust::device_vector<double> & d_vect_c,
               thrust::device_vector<double> & d_vect_d,
               thrust::device_vector<double> & d_vect_results){

    thrust::device_vector<double> d_vect_x(size,0.00),
        d_vect_a_prime(size,0.00),
        d_vect_c_prime(size,0.00),
        d_vect_temp(size,0.00);


//Define and create Cuda Streams

    cudaStream_t s1,s2,s3;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaStreamCreate(&s3);

//Foward Reduction Phase

    int level = 1;
    while(level < size){

    //AlphaBeta Methods
        d_vect_x = d_vect_d;

        LowerAlphaBeta(size,level,
            d_vect_a.data(),
            d_vect_a_prime.data(),
            d_vect_b.data()
        );

        UpperAlphaBeta(size, level,
            d_vect_b.data(),
            d_vect_c.data(),
            d_vect_c_prime.data()
        );


    //Front Methods

        MainFront(size, level,
            d_vect_a_prime.data(),
            d_vect_b.data(),
            d_vect_c.data(),
            d_vect_temp.data()
        );

        SolutionFront(size, level,
            d_vect_a_prime.data(),
            d_vect_d.data(),
            d_vect_x.data(),
            d_vect_temp.data()
        );

        LowerFront(size, level,
            d_vect_a.data(),
            d_vect_a_prime.data()
        );

    //Back Methods

        MainBack(size, level,
            d_vect_a.data(),
            d_vect_c_prime.data(),
            d_vect_b.data(),
            d_vect_temp.data()
        );

        SolutionBack(size, level,
            d_vect_c_prime.data(),
            d_vect_d.data(),
            d_vect_x.data(),
            d_vect_temp.data()
        );

        UpperBack(size, level,
            d_vect_c.data(),
            d_vect_c_prime.data()
        );

    //Set up diagonals for next reduction level
        d_vect_a = d_vect_a_prime;
        d_vect_c = d_vect_c_prime;
        d_vect_d = d_vect_x;

        level *= 2;
    }

//Backward Substitution Phase

        thrust::transform(
            d_vect_d.begin(), d_vect_d.end(),
            d_vect_b.begin(),
            d_vect_results.begin(),
            thrust::divides<double>()
        );



    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaStreamDestroy(s3);

    return ;
}


void LowerAlphaBeta(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_a_prime, DPtrD d_ptr_b){

//	thrust::transform(
//		d_ptr_a + level, d_ptr_a + n,
//		d_ptr_b,
//		d_ptr_a_prime + level,
//		AlphaBeta()
//	);
    int nn = n-level+1;
    dim3 block((nn)/128+ 1);

    point_alpha<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_a + level),
                             thrust::raw_pointer_cast(d_ptr_b),
                             thrust::raw_pointer_cast(d_ptr_a_prime + level),nn);

}

void UpperAlphaBeta(int n, int level, DPtrD d_ptr_b, DPtrD d_ptr_c, DPtrD d_ptr_c_prime){

//	thrust::transform(
//		d_ptr_c , d_ptr_c + (n-level),
//		d_ptr_b + level,
//		d_ptr_c_prime,
//		AlphaBeta()
//	);
    int nn = n-level+1;
    dim3 block((nn)/128+ 1);

    point_alpha<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_c ),
                             thrust::raw_pointer_cast(d_ptr_b + level),
                             thrust::raw_pointer_cast(d_ptr_c_prime),nn);
}

void MainFront(int n, int level, DPtrD d_ptr_a_prime, DPtrD d_ptr_b, DPtrD d_ptr_c, DPtrD d_ptr_temp){

//    thrust::transform(
//        d_ptr_a_prime + level, d_ptr_a_prime + n,
//        d_ptr_c,
//        d_ptr_temp,
//        thrust::multiplies<double>()
//    );
    int nn = n-level+1;
    dim3 block((nn)/128+ 1);

    // c = a*b
    // d_ptr_b = c + d_ptr_b
    point_mul_plus<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_a_prime+level ),
                             thrust::raw_pointer_cast(d_ptr_c ),
                             thrust::raw_pointer_cast(d_ptr_temp),
                                 thrust::raw_pointer_cast(d_ptr_b + level),nn);
//    thrust::transform(
//        d_ptr_b + level, d_ptr_b + n,
//        d_ptr_temp,
//        d_ptr_b + level,
//        thrust::plus<double>()
//    );

}

void SolutionFront(int n, int level, DPtrD d_ptr_a_prime, DPtrD d_ptr_d, DPtrD d_ptr_x, DPtrD d_ptr_temp ){

//    thrust::transform(
//        d_ptr_a_prime + level, d_ptr_a_prime + n,
//        d_ptr_d,
//        d_ptr_temp,
//        thrust::multiplies<double>()
//    );

//    thrust::transform(
//        d_ptr_x + level, d_ptr_x + n,
//        d_ptr_temp,
//        d_ptr_x + level,
//        thrust::plus<double>()
//    );

    int nn = n-level+1;
    dim3 block((nn)/128+ 1);

    // c = a*b
    // d_ptr_b = c + d_ptr_b
    point_mul_plus<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_a_prime + level ),
                             thrust::raw_pointer_cast(d_ptr_d ),
                             thrust::raw_pointer_cast(d_ptr_temp),
                                 thrust::raw_pointer_cast(d_ptr_x + level),nn);
}

void LowerFront(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_a_prime){

//    thrust::transform(
//        d_ptr_a_prime + level, d_ptr_a_prime + n,
//        d_ptr_a,
//        d_ptr_a_prime + level,
//        thrust::multiplies<double>()
//    );
    int nn = n-level+1;
    dim3 block((nn)/128+ 1);
    point_mul<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_a_prime + level ),
                             thrust::raw_pointer_cast(d_ptr_a),
                             thrust::raw_pointer_cast(d_ptr_a_prime + level),nn);
}

void MainBack(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_c_prime, DPtrD d_ptr_b, DPtrD d_ptr_temp){

//    thrust::transform(
//        d_ptr_c_prime , d_ptr_c_prime + (n - level),
//        d_ptr_a + level,
//        d_ptr_temp,
//        thrust::multiplies<double>()
//    );

//    thrust::transform(
//        d_ptr_b , d_ptr_b + (n - level),
//        d_ptr_temp,
//        d_ptr_b,
//        thrust::plus<double>()
//    );

    int nn = n-level+1;
    dim3 block((nn)/128+ 1);

    // c = a*b
    // d_ptr_b = c + d_ptr_b
    point_mul_plus<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_c_prime ),
                             thrust::raw_pointer_cast(d_ptr_a + level ),
                             thrust::raw_pointer_cast(d_ptr_temp),
                                 thrust::raw_pointer_cast(d_ptr_b),nn);
}

void SolutionBack(int n, int level, DPtrD d_ptr_c_prime, DPtrD d_ptr_d, DPtrD d_ptr_x, DPtrD d_ptr_temp){

//    thrust::transform(
//        d_ptr_c_prime, d_ptr_c_prime + (n-level),
//        d_ptr_d + level,
//        d_ptr_temp,
//        thrust::multiplies<double>()
//    );

//    thrust::transform(
//        d_ptr_x , d_ptr_x + (n-level),
//        d_ptr_temp,
//        d_ptr_x,
//        thrust::plus<double>()
//    );
    int nn = n-level+1;
    dim3 block((nn)/128+ 1);

    point_mul_plus<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_c_prime ),
                             thrust::raw_pointer_cast(d_ptr_d + level ),
                             thrust::raw_pointer_cast(d_ptr_temp),
                                 thrust::raw_pointer_cast(d_ptr_x),nn);
}


void UpperBack(int n, int level, DPtrD d_ptr_c, DPtrD d_ptr_c_prime){

//    thrust::transform(
//        d_ptr_c_prime, d_ptr_c_prime + (n-level),
//        d_ptr_c + level,
//        d_ptr_c_prime,
//        thrust::multiplies<double>()
//    );
    int nn = n-level+1;
    dim3 block((nn)/128+ 1);
    point_mul<<<block,128>>>(thrust::raw_pointer_cast(d_ptr_c_prime ),
                             thrust::raw_pointer_cast(d_ptr_c + level),
                             thrust::raw_pointer_cast(d_ptr_c_prime),nn);
}

}//END - namespace

#endif // BACKUP_H
