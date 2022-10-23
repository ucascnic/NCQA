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
 
#include <stdio.h>
using namespace arma;
#include "time.h"

#include<cuda_runtime_api.h>
#include<cublas_v2.h>
#include<stdio.h>
#include<stdlib.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}


void record_time(std::string & filestr,std::vector<double> &value);

static const double alpha_1   =  -1.0 ;
static const double alpha1  =  1.0 ;
static const double beta   =  0.0 ;

double getTime(time_t start_time)
{
    time_t end_time;
    time(&end_time);

    double seconds_total=difftime(end_time,start_time);

    return seconds_total;
}

void matrix_times(cublasHandle_t handle,double *f,double *g,int mf,int nf, int kg,double *res){

    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,mf,kg,nf,
                 &alpha1 ,f ,mf ,g ,nf ,&beta,res,mf);
}

void matrix_timesV(cublasHandle_t handle,double *f,double *g,int mf,int nf,double *res){

    cublasDgemv(handle, CUBLAS_OP_N, mf, nf, &alpha1,
                f, mf, g, 1, &beta, res,1);
}
void matrixT_timesV(cublasHandle_t handle,double *f,double *g,int mf,int nf,double *res){

    cublasDgemv(handle, CUBLAS_OP_T, mf, nf, &alpha1,
                f, mf, g, 1, &beta, res,1);
}
void matrix_timesT(cublasHandle_t handle,double *f,double *g,int mf,int nf, int kg,double *res){

    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,mf,kg,nf,
                 &alpha1 ,f,mf,g,kg,&beta,res,mf);
}

void matrix_AtimesAT(cublasHandle_t handle,double *f,int mf,int nf,double *res){

    cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,mf,mf,nf,
                 &alpha1 ,f,mf,f,mf,&beta,res,mf);
}
void matrix_ATtimesA(cublasHandle_t handle,double *f,int mf,int nf,double *res){

    cublasDgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,nf,nf,mf,
                 &alpha1 ,f,mf,f,mf,&beta,res,nf);
}
void matrixT(cublasHandle_t handle, double *P, double *Pt, int m, int n){
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &alpha1, P, m, &beta,
                Pt,n, Pt, n);

}

#include"resources.h"

double  testGrad(Resources<double>*resources, double *P, double *mu,double *A,double *b, double r, int n,double *x);
__global__ void rosenfunctionvalkernel(double *x,double *output, int nn);
int main(int argc, char **argv){



	int input = atoi(argv[1]);
     std::vector<int> ns = { input};

   // std::vector<int> ns = {10};

    std::vector<double> times;
    std::vector<double> values;
    int repeat = 10;
    srand(0);

    int maxsize =  ns[ns.size()-1];
    Resources<double> resources(maxsize*maxsize+20*maxsize);
    for (int kk = 0; kk < ns.size(); ++kk){

        size_t n = ns[kk];
		size_t m = 1;
		repeat = repeat - 1;

        for (int k = 0; k < repeat; k++){

        mat P_cpu_;
        P_cpu_.randu(n, n);

        mat P_cput = trans(P_cpu_) ;
        mat P_cpu = P_cpu_ *  P_cput   ;


		mat mu_cpu;
        mu_cpu.ones(n, 1);
		
        std::cout << k << std::endl;
        thrust::device_vector<double> P(P_cpu.memptr(),P_cpu.memptr()+n*n);
		thrust::device_vector<double> mu(mu_cpu.memptr(),mu_cpu.memptr()+n);

		thrust::device_vector<double> A(m*n,1.0);
		thrust::device_vector<double> b(m,1.0);




   
 
            double xx = 1.0 / ((double) n);
            thrust::device_vector<double> x(n,xx);
            clock_t start, finish;double duration;

            start = clock();
			

            double res = 0.0;
			double tol = 1e-6;
			if (n>4000) tol=1e-5;
            res = testGrad(&resources,thrust::raw_pointer_cast(P.data()),
                          thrust::raw_pointer_cast(mu.data()),
                                thrust::raw_pointer_cast(A.data()),
                                    thrust::raw_pointer_cast(b.data()),tol,n,thrust::raw_pointer_cast(x.data()));
					


            finish = clock();

            duration = (double)(finish - start) / CLOCKS_PER_SEC;

            values.push_back(res);
            times.push_back(duration);
        }
        FILE *fpvalues;
        FILE *fptimes;
        std::string val = std::to_string(n) + "_values_grad_generallized.txt";
        std::string tim = std::to_string(n) + "_times_grad_generallized.txt";
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
        printf("%.8f\t",res[i]);
    }

    printf("\n__________");
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

double rosenfunctionval(double *x, int n);

//x = y - t*g;
__global__ void update_x(double *x,double *y,double *g, double t, int nn){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=nn)
        return;
    x[i] =   y[i] - t*g[i];


}

__global__ void update_y_x(double *y,double *x,double *temp,   int nn){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=nn)
        return;
    temp[i] =   y[i] - x[i];


}

//__global__ void update_y_xall(double *y,double *x,double *temp,
//                           double *y2,double *x2,double *temp2,
//                              double *y3,double *x3,
//                              double *y4,double *x4,int nn){

//    int i = blockIdx.x * blockDim.x + threadIdx.x;
//    if (i>=nn)
//        return;
//    temp[i] =   y[i] - x[i];
//    temp2[i] =   y2[i] - x2[i];
//    y3[i] *= x3[i];
//    y4[i] *= x4[i];

//}

//y = x + (1-theta)*temp;
__global__ void update_y_xtemp(double *y,double *x,double *temp, double theta,  int nn){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=nn)
        return;
    y[i] =   x[i] + (1.0 - theta) * temp[i];

}


__global__ void update_yx(double *y,double *x,int nn){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=nn)
        return;
     y[i] += x[i];


}
__global__ void update_cst(double *cst, double *A,double  inv,int nn){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=nn)
        return;
     cst[i]  =    A[i]/inv;


}

__global__ void update_coeff(double *coeff, double *A,double  inv,int n,int nn){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i>=nn)
        return;

     coeff[i]  =    -1.0/inv;
     if (i%n == i/n){
        coeff[i] = 1.0 -1.0/inv;
     }

}
void prox_h(cublasHandle_t handle,double*coeff,  double *cst, double *x ,double t ,int n){
    matrix_timesV(handle,coeff,x,n,n,x);
    dim3 block((n)/128+ 1);
    update_yx<<<block,128>>>(x,cst,n);

}
void  grad_f(cublasHandle_t handle,double *g,double *P,double *x_hat,double * mu,int n){
	//opts.Q * x + opts.q
    matrix_timesV(handle,P,x_hat,n,n,g);
    dim3 block((n)/128+ 1);
    update_yx<<<block,128>>>(g,mu,n);

}
double  testGrad(Resources<double>*resources, double *P, double *mu, double *A,double *b, double r, int n,double * x){

    cublasHandle_t handle;
    cublasCreate(&handle);
    int k = 0;
    int maxiter = 1e7;
    resources->init();
    //Resources<double> resources(n*n + n*15);
    double * g = resources->allocate_resource(n);
    double * g_old = resources->allocate_resource(n);
    double * g_hat = resources->allocate_resource(n);
    double * temp = resources->allocate_resource(n);
    double * temp2 = resources->allocate_resource(n);
    double * x_old = resources->allocate_resource(n);
    double * y_old = resources->allocate_resource(n);
    double * x_hat = resources->allocate_resource(n);
    double * y = resources->allocate_resource(n);
    double * cst = resources->allocate_resource(n);
    double * coeff = resources->allocate_resource(n*n);
    cudaMemcpy(y,x,n*sizeof(double),cudaMemcpyDeviceToDevice); 
    double bb = (double)n;
    dim3 block((n)/128+ 1);
    update_cst<<<block,128>>>(cst,A,bb,n);

    dim3 blockn((n*n)/128+ 1);
    update_coeff<<<blockn,128>>>(coeff,A,bb,n,n*n);







	double theta = 1.0;

    grad_f(handle, g,P,y,mu,n);

	double BETA = 0.5;
    double err1 = 0.0;
	double t = 0.0;
	double t_hat = 0.0;
	
	double *y_x ;
	double *xnorm;
	double *x_x_old;
	double *restart;
	double *normg;
	cudaMallocHost((void**)&y_x,sizeof(double));
	cudaMallocHost((void**)&xnorm,sizeof(double));
	cudaMallocHost((void**)&x_x_old,sizeof(double));
	cudaMallocHost((void**)&restart,sizeof(double));
	cudaMallocHost((void**)&normg,sizeof(double));
	
	
	cublasDnrm2(handle,n,g,1,normg);
	
	
	t = 1.0 / *normg;
	//x_hat = x - t*g;
	update_x<<<block,128>>>(x_hat,x,g,t,n);
    grad_f(handle,g_hat,P,x_hat,mu,n);

    update_y_x<<<block,128>>>(g,g_hat,temp,n);
    update_y_x<<<block,128>>>(x,x_hat,temp2,n);


//    show_res(x_hat,n);

	clock_t start_kernel_1, finish_kernel_1;double duration_kernel_1;std::vector<double> time1;
	clock_t start_kernel_2, finish_kernel_2;double duration_kernel_2;std::vector<double> time2;
	clock_t start_kernel_3, finish_kernel_3;double duration_kernel_3;std::vector<double> time3;
	clock_t start_kernel_4, finish_kernel_4;double duration_kernel_4;std::vector<double> time4;


			
			
    cublasDdot(handle,n,temp,1,temp,1,normg);
	cublasDdot(handle,n,temp,1,temp2,1,restart); 

    t = abs(*restart) / (*normg );


//    std::cout << t << std::endl;exit(0);
//        std::cout << restart << std::endl;exit(0);
		
    while( k < maxiter){
		cudaMemcpy(x_old,x,n*sizeof(double),cudaMemcpyDeviceToDevice);
		cudaMemcpy(y_old,y,n*sizeof(double),cudaMemcpyDeviceToDevice);
		
		//x = y - t*g;



		update_x<<<block,128>>>(x,y,g,t,n);


		start_kernel_2 = clock();
        prox_h(handle,coeff,cst,x,t,n);            					
		// y - x
        update_y_x<<<block,128>>>(y,x,temp,n);
		finish_kernel_2 = clock();
		duration_kernel_2 = (double)(finish_kernel_2 - start_kernel_2)  ;
		time2.push_back(duration_kernel_2);

        if ( k % 10 == 0){
            cublasDdot(handle,n,temp,1,temp,1,y_x);
            cublasDdot(handle,n,x,1,x,1,xnorm);
    //		cublasDnrm2(handle,n,temp,1,y_x);
    //		cublasDnrm2(handle,n,x,1,xnorm);

            *xnorm = max(1.0,*xnorm);
            err1 = sqrt((*y_x)/(*xnorm));

            //show_res(x,n);

            printf("%.8f\n",err1);
            if (err1 < r)
            {
				printf("%.8f\n",err1);
                break ;
            }
        }


		theta = 2.0/(1.0 + sqrt(1.0+4.0/(theta*theta)));
		
		// x - x old
		start_kernel_4 = clock();
		update_y_x<<<block,128>>>(x,x_old,temp2,n);
		cublasDdot(handle,n,temp,1,temp2,1,restart);
		finish_kernel_4 = clock();
		duration_kernel_4 = (double)(finish_kernel_4 - start_kernel_4) ;
		time4.push_back(duration_kernel_4);		

        if ( *restart >= 0.0 ){
		
            cudaMemcpy(x,x_old,n*sizeof(double),cudaMemcpyDeviceToDevice);
			cudaMemcpy(y,x,n*sizeof(double),cudaMemcpyDeviceToDevice);
			theta = 1.0;
        }else{


            //y = x + (1-theta)*temp;
			
			start_kernel_1 = clock();
            update_y_xtemp<<<block,128>>>(y,x,temp2,theta,n);						
            finish_kernel_1 = clock();
            duration_kernel_1 = (double)(finish_kernel_1 - start_kernel_1)  ;
            time1.push_back(duration_kernel_1);

        }
		
		
		
		cudaMemcpy(g_old,g,n*sizeof(double),cudaMemcpyDeviceToDevice);
		
		
		start_kernel_3 = clock();
        grad_f(handle,g,P,y,mu,n);
		finish_kernel_3 = clock();
		duration_kernel_3 = (double)(finish_kernel_3 - start_kernel_3) ;
		time3.push_back(duration_kernel_3);



		
        update_y_x<<<block,128>>>(y,y_old,temp2,n);
        update_y_x<<<block,128>>>(g,g_old,temp,n);
        //cublasDnrm2(handle,n,temp2,1,x_x_old);
        cublasDdot(handle,n,temp2,1,temp2,1,x_x_old);
//		x_x_old = x_x_old * x_x_old;
        cublasDdot(handle,n,temp,1,temp2,1,restart);
		

        //update_y_xall<<<block,128>>>(y,y_old,g,g_old,temp,temp2,temp2,temp2,n);

		t_hat = 0.5*( *x_x_old /abs(*restart));
		t_hat = max( BETA*t, t_hat );
        t = min(t, t_hat);
		
		
		
		
        k = k + 1;
    }
    cublasDestroy(handle);
	
	
	// output time 1 time 2 time 3
	std::string t1 = "./output/" + std::to_string(n) + "_kernel1.txt";
	record_time(t1,time1);
 	std::string t2 = "./output/" + std::to_string(n) + "_kernel2.txt";
	record_time(t2,time2);
 	std::string t3 = "./output/" + std::to_string(n) + "_kernel3.txt";
	record_time(t3,time3);	
 	std::string t4 = "./output/" + std::to_string(n) + "_kernel4.txt";
	record_time(t4,time4);	
	
    return 1e6;

}
double  testGradbackup (Resources<double>*resources, double *P, double *mu, double *A,double *b, double r, int n,double * x){

    cublasHandle_t handle;
    cublasCreate(&handle);
    int k = 0;
    int maxiter = 1e7;
    resources->init();
    //Resources<double> resources(n*n + n*15);
    double * g = resources->allocate_resource(n);
    double * g_old = resources->allocate_resource(n);
    double * g_hat = resources->allocate_resource(n);
    double * temp = resources->allocate_resource(n);
    double * temp2 = resources->allocate_resource(n);
    double * x_old = resources->allocate_resource(n);
    double * y_old = resources->allocate_resource(n);
    double * x_hat = resources->allocate_resource(n);
    double * y = resources->allocate_resource(n);
    double * cst = resources->allocate_resource(n);
    double * coeff = resources->allocate_resource(n*n);
    cudaMemcpy(y,x,n*sizeof(double),cudaMemcpyDeviceToDevice); 
    double bb = (double)n;
    dim3 block((n)/128+ 1);
    update_cst<<<block,128>>>(cst,A,bb,n);

    dim3 blockn((n*n)/128+ 1);
    update_coeff<<<blockn,128>>>(coeff,A,bb,n,n*n);







	double theta = 1.0;

    grad_f(handle, g,P,y,mu,n);



    double err1 = 0.0;
	double y_x = 0.0;
	double xnorm = 0.0;
	double x_x_old = 0.0;
	double restart = 0.0;
	double BETA = 0.5;
	double t = 0.0;
	double t_hat = 0.0;
	
	double normg = 0.0;
	cublasDnrm2(handle,n,g,1,&normg);
	
	
	t = 1.0 / normg;
	//x_hat = x - t*g;
	update_x<<<block,128>>>(x_hat,x,g,t,n);
    grad_f(handle,g_hat,P,x_hat,mu,n);

    update_y_x<<<block,128>>>(g,g_hat,temp,n);
    update_y_x<<<block,128>>>(x,x_hat,temp2,n);


//    show_res(x_hat,n);

	clock_t start_kernel_1, finish_kernel_1;double duration_kernel_1;std::vector<double> time1;
	clock_t start_kernel_2, finish_kernel_2;double duration_kernel_2;std::vector<double> time2;
	clock_t start_kernel_3, finish_kernel_3;double duration_kernel_3;std::vector<double> time3;
	clock_t start_kernel_4, finish_kernel_4;double duration_kernel_4;std::vector<double> time4;


			
			
    cublasDdot(handle,n,temp,1,temp,1,&normg);
	cublasDdot(handle,n,temp,1,temp2,1,&restart); 

    t = abs(restart) / (normg );


//    std::cout << t << std::endl;exit(0);
//        std::cout << restart << std::endl;exit(0);
		
    while( k < maxiter){
		cudaMemcpy(x_old,x,n*sizeof(double),cudaMemcpyDeviceToDevice);
		cudaMemcpy(y_old,y,n*sizeof(double),cudaMemcpyDeviceToDevice);
		
		//x = y - t*g;



		update_x<<<block,128>>>(x,y,g,t,n);


		start_kernel_2 = clock();
        prox_h(handle,coeff,cst,x,t,n);            					
		// y - x
        update_y_x<<<block,128>>>(y,x,temp,n);
		finish_kernel_2 = clock();
		duration_kernel_2 = (double)(finish_kernel_2 - start_kernel_2) ;
		time2.push_back(duration_kernel_2);

        if ( k % 10 == 0){
            cublasDdot(handle,n,temp,1,temp,1,&y_x);
            cublasDdot(handle,n,x,1,x,1,&xnorm);
    //		cublasDnrm2(handle,n,temp,1,&y_x);
    //		cublasDnrm2(handle,n,x,1,&xnorm);

            xnorm = max(1.0,xnorm);
            err1 = sqrt(y_x/xnorm);

            //show_res(x,n);

            printf("%.8f\n",err1);
            if (err1 < r)
            {
				printf("%.8f\n",err1);
                break ;
            }
        }


		theta = 2.0/(1.0 + sqrt(1.0+4.0/(theta*theta)));
		
		// x - x old
		start_kernel_4 = clock();
		update_y_x<<<block,128>>>(x,x_old,temp2,n);
		cublasDdot(handle,n,temp,1,temp2,1,&restart);
		finish_kernel_4 = clock();
		duration_kernel_4 = (double)(finish_kernel_4 - start_kernel_4)  ;
		time4.push_back(duration_kernel_4);		

        if ( restart >= 0.0 ){
		
            cudaMemcpy(x,x_old,n*sizeof(double),cudaMemcpyDeviceToDevice);
			cudaMemcpy(y,x,n*sizeof(double),cudaMemcpyDeviceToDevice);
			theta = 1.0;
        }else{


            //y = x + (1-theta)*temp;
			
			start_kernel_1 = clock();
            update_y_xtemp<<<block,128>>>(y,x,temp2,theta,n);						
            finish_kernel_1 = clock();
            duration_kernel_1 = (double)(finish_kernel_1 - start_kernel_1) ;
            time1.push_back(duration_kernel_1);

        }
		
		
		
		cudaMemcpy(g_old,g,n*sizeof(double),cudaMemcpyDeviceToDevice);
		
		
		start_kernel_3 = clock();
        grad_f(handle,g,P,y,mu,n);
		finish_kernel_3 = clock();
		duration_kernel_3 = (double)(finish_kernel_3 - start_kernel_3)  ;
		time3.push_back(duration_kernel_3);



		
        update_y_x<<<block,128>>>(y,y_old,temp2,n);
        update_y_x<<<block,128>>>(g,g_old,temp,n);
        //cublasDnrm2(handle,n,temp2,1,&x_x_old);
        cublasDdot(handle,n,temp2,1,temp2,1,&x_x_old);
//		x_x_old = x_x_old * x_x_old;
        cublasDdot(handle,n,temp,1,temp2,1,&restart);
		

        //update_y_xall<<<block,128>>>(y,y_old,g,g_old,temp,temp2,temp2,temp2,n);

		t_hat = 0.5*( x_x_old /abs(restart));
		t_hat = max( BETA*t, t_hat );
        t = min(t, t_hat);
		
		
		
		
        k = k + 1;
    }
    cublasDestroy(handle);
	
	
	// output time 1 time 2 time 3
	std::string t1 = "./output/" + std::to_string(n) + "_kernel1.txt";
	record_time(t1,time1);
 	std::string t2 = "./output/" + std::to_string(n) + "_kernel2.txt";
	record_time(t2,time2);
 	std::string t3 = "./output/" + std::to_string(n) + "_kernel3.txt";
	record_time(t3,time3);	
 	std::string t4 = "./output/" + std::to_string(n) + "_kernel4.txt";
	record_time(t4,time4);	
	
    return 1e6;

}


void record_time(std::string & filestr,std::vector<double> &value){
	FILE *fpvalues;
	
	fpvalues = fopen(filestr.c_str(), "w+");
	for (int i = 0 ; i < value.size(); ++i){
		fprintf(fpvalues, "%.12f\n",value[i]);
	}
	fclose(fpvalues);
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

