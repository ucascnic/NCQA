#include<iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include<cuda_runtime_api.h>
#include <thrust/device_ptr.h>
#include <cstdlib>
#include <cublas_v2.h>
#define ARMA_ALLOW_FAKE_GCC
#include<armadillo>
#include<string>
#include "utils/utils.h"
#include "Cyclic/cu_cr_solver.h"
#include "serial_tSolver.h"
#include <stdio.h>
using namespace arma;
#include "time.h"

double getTime(time_t start_time)
{
    time_t end_time;
    time(&end_time);

    double seconds_total=difftime(end_time,start_time);

    return seconds_total;
}

double  solverRosen(double *x0, double r, int n);
__global__ void rosenfunctionvalkernel(double *x,double *output, int nn);
int main(int argc, char **argv){




//    std::vector<int> ns = {100,200,300,400,500,600,700,800,900,1000,1100,1200,
//                           1300,1400,1500,2000,2500,3000,4000,5000,10000,20000,40000,
//                           100000,200000};

    std::vector<int> ns = {400000};
    std::vector<double> times;
    std::vector<double> values;
    int repeat = 2;

    srand(time(NULL));

    for (int kk = 0; kk < ns.size(); ++kk){

        size_t n = ns[kk];
        thrust::device_vector<double> hostx(n);



        for (int k = 0; k < repeat; k++){
            for (int i  = 0 ;i < n; ++i){
               hostx[i] =  (double)rand()/(double)RAND_MAX;
               hostx[i] = (hostx[i]-0.5)*6.0;
            }
            thrust::device_vector<double> x(hostx);


            clock_t start, finish;double duration;
            start = clock();
            double res = solverRosen(thrust::raw_pointer_cast(x.data()),1e-6,n);
            finish = clock();

            duration = (double)(finish - start) / CLOCKS_PER_SEC;

            values.push_back(res);
            times.push_back(duration);
        }
        FILE *fpvalues;
        FILE *fptimes;
        std::string val = std::to_string(n) + "_values.txt";
        std::string tim = std::to_string(n) + "_times.txt";
        fpvalues = fopen(val.c_str(), "w+");
        fptimes = fopen(tim.c_str(), "w+");

        for (int i = 0 ; i < values.size(); ++i){
            fprintf(fpvalues, "%.12f\n",values[i]);
            fprintf(fptimes, "%.12f\n",times[i]);

        }
        fclose(fpvalues);
        fclose(fptimes);


    }

    return 0;

}



//int main2(int argc, char **argv){
//    size_t diagonal_size = 0;

//    if (argc < 2) {
//        return 0;
//    }
//    else
//    {
//        diagonal_size = atoi(argv[1]);
//    }
//    if (diagonal_size==0)
//        return 0;

//    PCR_Solver crs = PCR_Solver(diagonal_size);

//    //Generate sampel data
//    srand (time(NULL));


//    thrust::device_vector<double> alist(diagonal_size);
//    thrust::device_vector<double> blist(diagonal_size);
//    thrust::device_vector<double> clist(diagonal_size);
//    thrust::device_vector<double> dlist(diagonal_size);
//    thrust::device_vector<double> xlist(diagonal_size);

//    double * ptr_alist = thrust::raw_pointer_cast(alist.data());
//    double * ptr_blist = thrust::raw_pointer_cast(blist.data());
//    double * ptr_clist = thrust::raw_pointer_cast(clist.data());
//    double * ptr_dlist = thrust::raw_pointer_cast(dlist.data());
//    double * ptr_xlist = thrust::raw_pointer_cast(xlist.data());

//    for (int i=0; i < diagonal_size; i++) {
//        alist[i] = 0;
//        blist[i] = i+1;
//        clist[i] = 0;
//        dlist[i] =  10;//rand() % 100 + 1;
//        xlist[i] = 0.0f;
//    }

//    alist[0] = double(0.0);
//    clist[diagonal_size-1] = double(0.0);

//    crs.Solve(ptr_alist, ptr_blist, ptr_clist, ptr_dlist, ptr_xlist);

//    for (auto item : xlist) {
//        std::cout << item << std::endl;
//    }



//    return 0;

//}
__global__ void  updategf (double *g,double *f,double *a,double *diag,double *b,
                           double *x,double tau,int nn){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n>=nn)
        return;

    double y = -20.0 * x[n];
    double base2 =  2.0*(y*y +  100.0) + 2.0 + tau;

    diag[n] = base2;
    if (n < nn - 1){
        g[n] = 1.0 - x[n];
        f[n] = 10.0*(x[n+1]-x[n]*x[n]);
        a[n+1] = 20.0 * y;
        b[n] = 20.0 * y;
        if (n==0){
            diag[n]  = 2.0*( y*y) + 2.0 + tau;
        }
    }else{
        b[n] = 0.;
        diag[n] = 2.0*100.0 + tau;
    }





}
void show_res(double *s,int n){
    double *res = (double *)malloc(n*sizeof(double));
     (cudaMemcpy(res,s,n*sizeof(double),cudaMemcpyDeviceToHost));
    for (int i = 0 ; i< n;++i){
        printf("%.6f\t",res[i]);
    }

    printf("\n");
    free(res);
}
__global__ void  updatex(double *deltax,double *xhat,
                         double *x0,double gamma_,int nn){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n>=nn)
        return;

    deltax[n] =  xhat[n] - x0[n];
    x0[n] = x0[n] +  gamma_ * (deltax[n]);

}
__global__ void  updateqk(double *qk, double *f,
                                         double *g,
                                         double *x0,
                                         double *a,
                                         double *diag,
                                         double *b,
                                         int nn){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n>=nn)
        return;
    // qk = Q * x -  2 * gradff - 2 * gradgg
    double a1,b1,c1;
    c1 = -g[n];
    if (n == nn-1)
        c1 = 0.0;

    if ((n > 0) && (n < nn-1)){
        b1 = f[n-1]*10.0 + (-20 * x0[n] * f[n]);
        a1 = x0[n-1]*a[n] +  x0[n]*diag[n] + x0[n+1]*b[n];
    }
    else{
        if (n ==0){
            b1 = -20 * x0[n] * f[n];
            a1 =   x0[n]*diag[n] + x0[n+1]*b[n];
        }
        else{

            b1 = f[n-1]*10.0;  // nn-1
            a1 =  x0[n-1]*a[n] +  x0[n]*diag[n] ;
        }

    }
    qk[n] = a1 -  2 * b1 - 2 * c1;








}
#include<resources.h>
double rosenfunctionval(double *x, int n);
double  solverRosen(double *x0, double r, int n){

    cublasHandle_t handle;cublasCreate(&handle);
    int k = 0;
    int maxiter = 1e7;
    Resources<double> resources(n*12);
    double * deltax = resources.allocate_resource(n);
    //double * xhatt = resources.allocate_resource(n);
    double * g = resources.allocate_resource(n);
    double * f = resources.allocate_resource(n);
    double * a = resources.allocate_resource(n);
    double * b = resources.allocate_resource(n);
    double * diag = resources.allocate_resource(n);
    double * xhat = resources.allocate_resource(n);
    double * qk = resources.allocate_resource(n);
    double gamma_ = 1.0;
    double zeta = 1e-3;
    double tau = 1.0;
    dim3 block((n)/128+ 1);
    int size = n;
    thrust::device_vector<double> d_vect_x(size,0.00),
        d_vect_a_prime(size,0.00),
        d_vect_c_prime(size,0.00),
        d_vect_temp(size,0.00);

    typedef thrust::device_vector<double> DVectorD;
    while( k < maxiter){



        updategf<<<block,128>>>(g,f,a,diag,b,x0,tau,n);
        updateqk<<<block,128>>>(qk,f,g,x0,a,diag,b,n);

        gamma_ =   gamma_ * (1.0 -  zeta *  gamma_);



        cyclic_reduction::Solve2(n,
                        a,
                        diag,
                        b,
                        qk,
                        xhat,d_vect_x,d_vect_a_prime,
                                 d_vect_c_prime,d_vect_temp
                    );

        updatex<<<block,128>>>(deltax,xhat,x0,gamma_,n);

        if (k%100 == 0){
            double norm_residual = 0.0;
            cublasDnrm2(handle,n,deltax,1,&norm_residual);
            printf("norm residual =  %.10f\n",norm_residual);
            if (norm_residual < r){
                double res = 0.0;
                dim3 block((n-1)/128+1);
                rosenfunctionvalkernel<<<block,128>>>(x0,xhat,n-1);
                cublasDasum(handle,n-1,xhat,1,&res);
                return res;
            }

        }
        k = k + 1;
    }
    return 1e6;

}
__global__ void rosenfunctionvalkernel(double *x,double *output, int nn){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=nn)
        return;

    output[i] =  (1-x[i])*(1-x[i]) + 100*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]);


}
double rosenfunctionval(double *x, int n){
    double res = 0.0;
    for (int i=0; i < n-1; ++i){
        res +=  (1-x[i])*(1-x[i]) + 100*(x[i+1]-x[i]*x[i])*(x[i+1]-x[i]*x[i]);
    }
    return res;
}
