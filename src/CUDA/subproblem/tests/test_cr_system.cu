#include "gtest/gtest.h"
#include <cyclic-reduction/cu_cr_solver.h>
#include <utils/utils.h>
#include <cmath>

#define CRSTEST(name) \
	TEST_F(CyclicReductionSolverTest,name)

/**
*	This test file holds all the tests for black box testing methods for the cyclic-reduction method
* 	of solving tridiagonal matricies
**/

using namespace cyclic_reduction;

class CyclicReductionSolverTest : public ::testing::Test{

protected:
//SetUp Methods
	static void SetUpTestCase(){
	}

	virtual void SetUp(){
	}
	
//Protected Data Members
	int n;
	std::string name;	

	//Host vectors for the matrix diagonals of matrix T and the column matrix D
	HVectorD h_vect_a,
		h_vect_b,
		h_vect_c,
		h_vect_d;


	//Host vectors for the results of the method calls
	HVectorD h_vect_results_e; //Expected result
	HVectorD h_vect_results_a; //Actual result


//Protected Methods
	
	void InitTest(std::string test_name, int n_){
		name = test_name;
		n = n_;
		h_vect_results_e.resize(n);
		h_vect_results_a.resize(n);
		h_vect_a.resize(n);
		h_vect_b.resize(n);
		h_vect_c.resize(n);
		h_vect_d.resize(n);
	}

	void Test(){
		LogVectors();
	
		h_vect_results_a = Solve(n,
					h_vect_a,
					h_vect_b,
					h_vect_c,	
					h_vect_d
				);
			
		LogResults();
		CheckResults();
	}

	void LogResults(){
		LogVector(name+":Actual",h_vect_results_a);
		LogVector(name+":Expected",h_vect_results_e);
	}

	void CheckResults(){
		for(int i = 0; i < n; i++){
			EXPECT_TRUE(CompareValues(h_vect_results_e[i], h_vect_results_a[i]))
				<< "For test "
				<< name 
				<< " Expected and actual elements differ by too much at index " 
				<< i;
		}
	}

	
	bool CompareValues(double expected, double actual){
		if(fabs(actual - expected) < 0.001){
			return true;
		} else{
			return false;
		}
	}


	void LogVectors(){
		LogVector(name+":A",h_vect_a);	
		LogVector(name+":B",h_vect_b);
		LogVector(name+":C",h_vect_c);
		LogVector(name+":D",h_vect_d);
	
	}	

	void LogVector(std::string name, HVectorD vector){
		utils::PrintVector(true,
			 name,
			vector 
		);
	}
	


};


/*
*           T   *  X   =  D      
* | 2 1 0 0 0 |   |x1|   | 2|            | 0.45161290322581|
* | 2 3 2 0 0 | * |x2| = | 4|   ==> X ~= | 1.09677419354838|
* | 0 3 4 3 0 |   |x3|   | 6|            |-0.09677419354838|
* | 0 0 4 5 0 |   |x4|   | 8|            | 1.03225806451612|
* | 0 0 0 5 6 |   |x5|   |10|		 | 0.80645161290323|
*/


CRSTEST(GeneralCase1){
	InitTest("GeneralCase1",5);
	
	for(int i=0; i<n; i++){
		h_vect_a[i] = i+1;
		h_vect_b[i] = i+2;
		h_vect_c[i] = i+1;
		h_vect_d[i] = (i+1)*2;	
	}
	
	h_vect_results_e[0] = 0.45161290322581;
	h_vect_results_e[1] = 1.096774193548387;
	h_vect_results_e[2] = -0.096774193548387;
	h_vect_results_e[3] = 1.032258064516129;
	h_vect_results_e[4] = 0.80645161290323; 
		
	Test();	
}


CRSTEST(GeneralCase2){
	InitTest("GeneralCase2",10);
	
	for(int i=0; i<n; i++){
		h_vect_a[i] = i+40.2;
		h_vect_b[i] = i+351.345;
		h_vect_c[i] = i* 34.523;
		h_vect_d[i] = (i+34.234)*2;	
	}
	
	h_vect_results_e[0] = 0.19487398426048;
	h_vect_results_e[1] = 0.16192025637815;
	h_vect_results_e[2] = 0.15605246111026;
	h_vect_results_e[3] = 0.15199444297771;
	h_vect_results_e[4] = 0.13390167697961;
	h_vect_results_e[5] = 0.16053467408732;
	h_vect_results_e[6] = 0.088114681591252;
	h_vect_results_e[7] = 0.20065829139964;
	h_vect_results_e[8] = 0.026500310754682;
	h_vect_results_e[9] = 0.23634068659443; 
		
	Test();	
}





