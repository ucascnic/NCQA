#include "cu_functors.cu"
#include <thrust/host_vector.h>
#include <iostream>
using namespace std;
using namespace thrust;

host_vector<double> serial_solve(double n, host_vector<double> a, host_vector<double> b, host_vector<double> c, host_vector<double> d)
{

	Hydrogen hydroObj;
	Helium heliObj;

	n--;
	c[0] /= b[0];
	d[0] /= b[0];

	for(int i = 1; i < n; i++)
	{
		c[i] = hydroObj(a[i], b[i], c[i], c[i-1]);
		d[i] = heliObj(a[i], b[i], d[i], c[i-1], d[i-1]);
	}
 
	d[n] = heliObj(a[n], b[n], d[n], c[n-1], d[n-1]);


	Silicon siliObj;
	for(double i = n; i-- > 0;)
	{
		 d[i] = siliObj(c[i], d[i], d[i+1]);		
	}

	return d;
}

/*
int main(){
	double n = 4;
	host_vector<double> a(4);
	a[0] = 0.0; a[1] = -1.0; a[2] = -1.0; a[3] = -1.0;
	host_vector<double> b(4);
	b[0] = 4.0; b[1] = 4.0; b[2] = 4.0; b[3] = 4.0;
	host_vector<double> c(4);
	c[0] = -1.0; c[1] = -1.0; c[2] = -1.0; c[3] = 0.0;
	host_vector<double> d(4);
	d[0] = 5; d[1] = 5; d[2] = 10; d[3] = 23.0;
	// results { 2, 3, 5, 7}

	host_vector<double> result = serial_solve(n,a,b,c,d);
	for (int i = 0; i < n; i++)
	{
		cout << result[i] << endl;
	}

	return 0;
}*/
