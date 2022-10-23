// serial_tSolver.h
// David Walker

#ifndef SERIAL_TSOLVER_H
#define SERIAL_TSOLVER_H

#include <thrust/host_vector.h>
using namespace thrust;

host_vector<double> serial_solve(double n, host_vector<double> a, host_vector<double> b, host_vector<double> c, host_vector<double> d);

#endif

