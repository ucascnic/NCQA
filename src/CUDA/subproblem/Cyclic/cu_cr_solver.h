#ifndef CU_CR_SOLVER_H
#define CU_CR_SOLVER_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
namespace cyclic_reduction{

/**
 * TODO testing
 * This method will perform the cyclic reduction method of solving tridiagaonl matrices.
 * 
 * Parameters:
 * - size - The dimmension of the matrix
 * - vect_a - The sub diagonal of the matrix
 * - vect_b - The main diagonal of the matrix
 * - vect_c - The super diagonal of the matrix
 * - vect_d - The result column AX=D
 *
 * Returns:
 * - Vector containing the resuls 
 */
void Solve(int size,
        double *vect_a,
        double *vect_b,
        double *vect_c,
        double *vect_d,
        double *h_vect_results
    );


} //END - namespace

thrust::host_vector<double>  crSolve(int size, thrust::host_vector<double> vect_a, thrust::host_vector<double> vect_b, thrust::host_vector<double> vect_c, thrust::host_vector<double> vect_d);




#endif

