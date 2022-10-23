#include "gtest/gtest.h"

#include "test_functors.cu"
#include "test_cr.cu"
#include "test_serial_tSolver.cu"
#include "test_cr_system.cu"

int main(int argc, char **argv){
	::testing::InitGoogleTest(&argc,argv);
	return RUN_ALL_TESTS();
}
