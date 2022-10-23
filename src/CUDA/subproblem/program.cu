#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "utils/utils.h"
#include "Cyclic/cu_cr_solver.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "serial_tSolver.h"

using namespace std;

void usage(){

    cout << "usage: ./program [option]"
        << endl << endl
        << "p \t run parallel method" << endl
        << "s \t run serial method" << endl;
}

int main(int argc, const char *argv[]){
    typedef thrust::device_vector<double> DVectorD;

    char opt;





    string line;
    ifstream myfile ("input.txt");
    if (myfile.is_open())
    {
        getline (myfile, line);
        int n;
        stringstream stream;
        stream <<line;
        stream >>n;

        getline (myfile, line);
        int a;
        vector<int> A;
        stringstream stream_a;
        stream_a <<line;
        for (int i=0; i<n; i++){
            stream_a >>a;
            A.push_back(a);}

        getline (myfile, line);
        int b;
        vector<int> B;
        stringstream stream_b;
        stream_b <<line;
        for (int i=0; i<n; i++){
            stream_b >>b;
            B.push_back(b);}

        getline (myfile, line);
        int c;
        vector<int> C;
        stringstream stream_c;
        stream_c <<line;
        for (int i=0; i<n; i++){
            stream_c >>c;
            C.push_back(c);}

        getline (myfile, line);
        int d;
        vector<int> D;
        stringstream stream_d;
        stream_d <<line;
        for (int i=0; i<n; i++){
            stream_d >>d;
            D.push_back(d);}


        myfile.close();


        for (int i = 0 ; i< A.size(); ++i)
            std::cout << A[i] << "\t";
        std::cout <<   std::endl;
        for (int i = 0 ; i< B.size(); ++i)
            std::cout << B[i] << "\t";
                std::cout <<   std::endl;
        for (int i = 0 ; i< C.size(); ++i)
            std::cout << C[i] << "\t";
                std::cout <<   std::endl;
        for (int i = 0 ; i< D.size(); ++i)
            std::cout << D[i] << "\t";
                std::cout <<   std::endl;
        DVectorD h_vect_a = A;
        DVectorD h_vect_b = B;
        DVectorD h_vect_c = C;
        DVectorD h_vect_d = D;
        DVectorD h_vect_results(n);

        cyclic_reduction::Solve(n,
                        h_vect_a,
                        h_vect_b,
                        h_vect_c,
                        h_vect_d,
                        h_vect_results
                    );

        utils::LogProgramResults("Cyclic Reduction Method Results",h_vect_results);


    }
    else cout <<"Unable to open file";

    return 0;
}
