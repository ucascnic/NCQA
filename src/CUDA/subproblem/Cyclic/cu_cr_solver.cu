#include "cu_cr_solver.h"
#include "cu_cr_internal.h"
#include <cu_cr_functors.cu>

#include <cuda.h>
#include <math.h>
#include <thread>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>

/*
* For method documentation see cu_cr_internal.h unless otherwise specified.
*/

namespace cyclic_reduction{

void Solve(int size, double *d_vect_a,
               double * d_vect_b,
               double * d_vect_c,
               double * d_vect_d,
               double * d_vect_results){

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

	//Set up diagonals for next reduction level
        //d_vect_a = d_vect_a_prime;
        cudaMemcpy(d_vect_a,thrust::raw_pointer_cast(d_vect_a_prime.data()),
                   sizeof(double)*size,cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_vect_c,thrust::raw_pointer_cast(d_vect_c_prime.data()),
                   sizeof(double)*size,cudaMemcpyDeviceToDevice);
        //d_vect_c = d_vect_c_prime;
        //d_vect_d = d_vect_x;
        cudaMemcpy(d_vect_d,thrust::raw_pointer_cast(d_vect_x.data()),
                   sizeof(double)*size,cudaMemcpyDeviceToDevice);
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

	thrust::transform(
		d_ptr_a + level, d_ptr_a + n,
		d_ptr_b,
		d_ptr_a_prime + level,
		AlphaBeta()
	);
		
}

void UpperAlphaBeta(int n, int level, DPtrD d_ptr_b, DPtrD d_ptr_c, DPtrD d_ptr_c_prime){

	thrust::transform(
		d_ptr_c , d_ptr_c + (n-level),
		d_ptr_b + level,
		d_ptr_c_prime,
		AlphaBeta()
	);

}

void MainFront(int n, int level, DPtrD d_ptr_a_prime, DPtrD d_ptr_b, DPtrD d_ptr_c, DPtrD d_ptr_temp){

	thrust::transform(
		d_ptr_a_prime + level, d_ptr_a_prime + n,
		d_ptr_c,
		d_ptr_temp,
		thrust::multiplies<double>()
	);

	thrust::transform(
		d_ptr_b + level, d_ptr_b + n,
		d_ptr_temp,
		d_ptr_b + level,
		thrust::plus<double>()
	);

}

void SolutionFront(int n, int level, DPtrD d_ptr_a_prime, DPtrD d_ptr_d, DPtrD d_ptr_x, DPtrD d_ptr_temp ){

	thrust::transform(
		d_ptr_a_prime + level, d_ptr_a_prime + n,
		d_ptr_d,
		d_ptr_temp,
		thrust::multiplies<double>()
	);

	thrust::transform(
		d_ptr_x + level, d_ptr_x + n,
		d_ptr_temp,
		d_ptr_x + level,
		thrust::plus<double>()
	);

}

void LowerFront(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_a_prime){

	thrust::transform(
		d_ptr_a_prime + level, d_ptr_a_prime + n,
		d_ptr_a,
		d_ptr_a_prime + level,
		thrust::multiplies<double>()
	);	
}

void MainBack(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_c_prime, DPtrD d_ptr_b, DPtrD d_ptr_temp){

	thrust::transform(
		d_ptr_c_prime , d_ptr_c_prime + (n - level),
		d_ptr_a + level,
		d_ptr_temp,
		thrust::multiplies<double>()
	);

	thrust::transform(
		d_ptr_b , d_ptr_b + (n - level),
		d_ptr_temp,
		d_ptr_b,
		thrust::plus<double>()
	);
}

void SolutionBack(int n, int level, DPtrD d_ptr_c_prime, DPtrD d_ptr_d, DPtrD d_ptr_x, DPtrD d_ptr_temp){
	
	thrust::transform(
		d_ptr_c_prime, d_ptr_c_prime + (n-level),
		d_ptr_d + level,
		d_ptr_temp,
		thrust::multiplies<double>()
	);

	thrust::transform(
		d_ptr_x , d_ptr_x + (n-level),
		d_ptr_temp,
		d_ptr_x,
		thrust::plus<double>()
	);

}


void UpperBack(int n, int level, DPtrD d_ptr_c, DPtrD d_ptr_c_prime){

	thrust::transform(
		d_ptr_c_prime, d_ptr_c_prime + (n-level),
		d_ptr_c + level,
		d_ptr_c_prime,
		thrust::multiplies<double>()
	);	
}

}//END - namespace
