#include "PCR_Class.h"
#include "PCR_Device_functions.h"


int count_iter(int bin) {

	int m = 0;
	int count = 0;

	while (bin != 0) {
		m+=bin%2;
		bin = bin >> 1;
		count++;
	}
	count--;

	if (m>1) m = 0;

    return (count - m);
}

PCR_Solver::PCR_Solver(int coming_ds) {

    diagonal_size = coming_ds;

    iter_max = count_iter(coming_ds);

}


void PCR_Solver::Solve(float * alist, float * blist, float * clist, float * dlist, float * xlist) {

    dim3 block((diagonal_size)/128+ 1);
    Solve_Kernel<<<block,128>>>(alist, blist, clist, dlist, xlist, iter_max, diagonal_size);

}

void PCR_Solver::Solve(double * alist, double * blist, double * clist, double * dlist, double * xlist) {

    dim3 block((diagonal_size)/128+ 1);
    Solve_KernelD<<<block,128>>>(alist, blist, clist, dlist, xlist, iter_max, diagonal_size);

}
