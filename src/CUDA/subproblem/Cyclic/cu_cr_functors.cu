#include <cuda.h>

namespace cyclic_reduction{
/**
* Calculates:
*	alpha_i = -a_i/b_(i-2^(l-i))
*	where:
*		x = a_i
*		y = b_(i-2^(l-i))
*
*	beta_i = -c_i/b_(i+2^(l-i))
*	where:
*		x = c_i
*		y = b_(i+2^(l-i))	
**/
struct AlphaBeta{
	__host__ __device__
	double operator()(double x, double y){
		if(y == 0.00){
			return 0.00;
		} else{
			return (-x)/y;
		}
	}
};





struct FirstStepFunctor{
	
	__host__ __device__
	double operator()(double a, double b){
		if(b == 0.00){
			return 0.00;
		} else{
			return -a/b;
		}	
	}
};

struct SecondStepFunctor{
	__host__ __device__
	double operator()(double c, double b){
		if(b == 0.00){
			return 0.00;
		}else{
			return -c/b;
		}
	}
};	

}





/**
* Calulates:
*	a_i = alpha_u * a_(i-2^(l-1))_prime
*	where:
*		x,y = as in AlphaBeta
*		z = a_(i-2^(l-1))_prime
*
*	c_i = beta_i * c_(i+2^(l-1))_prime
*	where:
*		x,y = as in AlphaBeta
*		z = c_(i+2^(l-1))_prime
**/
struct AC{
	__host__ __device__
	double operator()(double x, double y, double z){
		cyclic_reduction::AlphaBeta alphaBeta;
		return alphaBeta(x,y)*z;
	}
};



/**
* Calculates:
*	b_i = b_i_prime + alpha_i * c_(i-2^(l-1))_prime + beta_i * a_(i+2^(l-1))_prime
* 	where:
*		alpha_i , beta_i are results from AlphaBeta
*		x = b_i_prime
* 		y = c_(i-2^(l-1))_prime
*		z = (a_(i+2^(l-1))_prime
*
* 	d_i = d_i_prime + alpha_i * d_(i-2^(l-1))_prime + beta_i * d_(i+2^(l-1))_prime
* 	where:
*		alpha_i , beta_i are results from AlphaBeta
*		x = d_i_prime
* 		y = d_(i-2^(l-1))_prime
*		z = (d_(i+2^(l-1))_prime
*
*		
**/
struct BD{
	__host__ __device__
	double operator()(double x, double alpha, double y, double beta, double z){
		double a = alpha*y;
		double b = beta*z;
		return x + a + b;
	}
};

