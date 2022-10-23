#include "gtest/gtest.h"
#include "cu_functors.cu"
//#include "./cyclic-reduction/cu_cr_functors.cu"
//#include <cyclic-reduction/cu_cr_functors.cu>

#define CRFTEST(name) \
	TEST(CyclicReductionFunctorTest,name)

//Functor tests go here
TEST( FunctorTest, Hydrogen){
	Hydrogen hydrogen;
	EXPECT_EQ(-1.5,hydrogen(1.0,2.0,3.0,4.0));
}


TEST( FunctorTest, Helium){
	Helium helium;
	EXPECT_EQ(1 , helium(1,2,3,4,5));
}


TEST( FunctorTest, Silicon){
	Silicon silicon;
	EXPECT_EQ(-1 ,silicon(1,2,3));
}


/*

TEST( FunctorTest, AC){
	AC ac;
	EXPECT_EQ(-1.5 ,ac(1,2,3));
}

TEST( FunctorTest, BD){
	BD bd;
	EXPECT_EQ(27 , bd(1,2,3,4,5));
}
*/

/*
*	Cyclic Reduction Functor Tests
*/

/*

CRFTEST( AlphaBeta){
	cyclic_reduction::AlphaBeta alphaBeta;
	EXPECT_EQ(-2 ,alphaBeta(4,2));
	EXPECT_EQ(0.00,alphaBeta(4,0.00));
}

//TODO add test for when b is not equal to 0.00
CRFTEST( FirstStepFunctor){
	cyclic_reduction::FirstStepFunctor functor;
	EXPECT_EQ(0.00,functor(1.00,0.00));
}
*/
