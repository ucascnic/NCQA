#ifndef CU_CR_INTERNAL_H
#define CU_CR_INTERNAL_H

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>

/*
 * 	This header file contains all foward declarations of methods used internall by
 * 	cu_cr_solver.cu
 *	This file also serves as an interface to be used by testing.
 *	For method documentation see method declarations below.
 *
 *	Common parameters used by methods:
 *		n Size of diagonal
 *		level Reduction level
 *
 *		d_ptr_a Pointer to first element of lower diagonal device vector
 *		d_ptr_a_prime Pointer to first element of temp lower diagonal device vector
 *
 *		d_ptr_b Pointer to first element of main diagonal device vector
 *		d_ptr_b_prime Pointer to first element of temp main diagonal device vector
 *
 *		d_ptr_c Pointer to first element of upper diagonal device vector
 *		d_ptr_c_prime Pointer to first element of temp upper diagonal device vector
 *
 *		d_ptr_d Pointer to first element of column vector of right hand side of the equation Ax=B
 *
 *		d_ptr_x Pointer to first element of column vector containing solutions
 *
 */

int calc_q(int n);

void calc_init(int n,
	thrust::device_ptr<double> d_vect_a, 
	thrust::device_ptr<double> d_vect_b, 
	thrust::device_ptr<double> d_vect_c,
	thrust::device_ptr<double> d_vect_d);


namespace cyclic_reduction{

/*
 * 	TYPEDEFS
 */
	typedef thrust::device_ptr<double> DPtrD; //Device Ptr Double 
	typedef thrust::device_vector<double> DVectorD; //Device Vector Double
	typedef thrust::host_vector<double> HVectorD; //Host Vector Double
	typedef thrust::tuple<int, double> TupleID; //Tuple Integer Double
	typedef thrust::tuple<thrust::counting_iterator<int>, DVectorD::iterator> TupleCiDvi; //Tuple Counting iterator Device vector iterator
	typedef thrust::zip_iterator<TupleCiDvi> ZipIteratorTCD; //Zip Iterator TupleCiDvi

/*
 * 	FOWARD DECLARATIONS
 */

//Calculation Methods - In order of definition in cu_cr_solver.cu
	
	/*
  	 * Modifies d_ptr_a_prime
 	 *
	 * Performs AlphaBeta calculation for all elements in the lower diagonal where the 
	 * element position - reduction level >= 0.
	 * Stores the calculations in d_ptr_a_prime.
	 */
	void LowerAlphaBeta(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_a_prime, DPtrD d_ptr_b);

	/*
	 * Modifies d_ptr_c_prime
	 *
	 * Performs AlphaBeta calculation for all elements in the upper diagonal where the
	 * element position + reduction level < size of diagonal.
	 * Stores the calculations in d_ptr_c_prime.
	 */	
	void UpperAlphaBeta(int n, int level, DPtrD d_ptr_b, DPtrD d_ptr_c, DPtrD d_ptr_c_prime);
	
	/*
	 * Modifies d_ptr_d
	 *
 	 * Adds the result of multiplying the AlphaBeta calculation of the element at the same position
 	 * in the lower diagonal with neighboring upper element to qualifying main diagonal elements.
 	 * Stores result at position of element in the main diagonal.
 	 */	
	void MainFront(int n, int level, DPtrD d_ptr_a_prime, DPtrD d_ptr_b, DPtrD d_ptr_c, DPtrD d_ptr_temp);

	/*
	 * Modifies d_ptr_x
	 *
	 * Adds the result of multiplying the AlphaBeta calculation of the element at the same position
	 * in the lower diagonal with neighboring element in the right side column to the qualifying
	 * solution column elements.
	 */
	void SolutionFront(int n, int level, DPtrD d_ptr_a_prime, DPtrD d_ptr_d, DPtrD d_ptr_x, DPtrD d_ptr_temp);

	/*
 	 * Modifies d_ptr_a_prime
 	 *
 	 * Multiples each qualifying element in the lower diagonal by its AlphaBeta calculation
 	 */
	void LowerFront(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_a_prime);


	/*
	 * Modifies d_ptr_c
	 *
	 * Multiples each qualifying element in the main diagonal by the AlphaBeta calculation of the 
	 * element at the same position in the upper diagonal with neighboring lower element.
	 * Then adds the result of the multiplicaiton to the value of the element in the main diagonal.
	 * Stores result at position of element in the main diagonal.
	 */	
	void MainBack(int n, int level, DPtrD d_ptr_a, DPtrD d_ptr_c_prime, DPtrD d_ptr_b, DPtrD d_ptr_temp);

	/*
	 * Modifies d_ptr_x
	 *
	 * Adds the result of multiplying the AlphaBeta calculation of the element at the same position in the 
	 * upper diagonal with neighboring element in the right side column to the qualifying solution
	 * column elements.
	 */
	void SolutionBack(int n, int level, DPtrD d_ptr_c_prime, DPtrD d_ptr_d, DPtrD d_ptr_x, DPtrD d_ptr_temp);

	
	/*
	 * Modifies d_ptr_c_prime
	 *
	 * Multiples AlphaBeta calculation of qualifying elements in upper diagonal by neighbor element in 
	 * upper diagonal.
	 */
	void UpperBack(int n, int level, DPtrD d_ptr_c, DPtrD d_ptr_c_prime);



//Utility Methods

	/*
	 * Modifies d_ptr
	 * Fills vector with 0.00
	 *
	 * Params:
	 * 	d_ptr Device pointer to first element in vector to be filled
	 */
	void InitDPtrD(int n, DPtrD d_ptr);

	/*
	 * Modifies d_ptr_x
	 * Copies all the elements from the the right side column into the solution column
	 */
	void InitSolutionDPtrD(int n, DPtrD d_ptr_d, DPtrD d_ptr_x);
	

}


#endif
