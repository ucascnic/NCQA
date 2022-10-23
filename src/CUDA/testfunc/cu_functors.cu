#include <cuda.h>


/**
* This file contains all the functors used for triSolver

x_prime means that list is [i-1]

a => a[i]
a_prime => a[i-1]
**/



/**
* returns c/(b-a*c_prime)
**/
struct Hydrogen{
	__host__ __device__
	double operator()(double a, double b, double c , double c_prime){
		double x = a*c_prime;
		double y = b - x;
		return c/y;			
	}

};


struct Helium{
	__host__ __device__
	double operator()(double a, double b, double d, double c_prime, double d_prime){
		double x = a * d_prime;
		double y = a * c_prime;
		double A = d - x;
		double B = b - y;
		return A/B;
	}

};



struct Silicon{
	//d_prime => d[i+1]
	__host__ __device__
	double operator()(double c, double d, double d_prime){
		double x = c * d_prime;
		return d - x;
	}

};
