#ifndef PCR_CLASS_H
#define PCR_CLASS_H

#include <vector>

class PCR_Solver {
private:
    size_t diagonal_size;
    int iter_max;
    std::vector<int> sdlist;

public:
    PCR_Solver(int coming_ds);
    void Solve(float * alist, float * blist, float * clist, float * dlist, float * xlist);
    void Solve(double * alist, double * blist, double * clist, double * dlist, double * xlist);

};


#endif
