#ifndef RESOURCES_H
#define RESOURCES_H
#include<cuda_runtime_api.h>

template <typename T>
class Resources
{
public:
    int max_resources;
    int cnt;
    T *resources;
public:

    Resources(int n){
        this->max_resources = n;
        this->cnt = 0;
        cudaMalloc((void**)&this->resources,sizeof(T) * this->max_resources);
        cudaMemset(this->resources,0,sizeof(T) * this->max_resources);
    }
    void init(){
        this->cnt = 0;
        cudaMemset(this->resources,0,sizeof(T) * this->max_resources);
    }
    ~Resources( ){
        cudaFree(this->resources);
     }

    T* allocate_resource(int required){
        int nn = this->cnt;
        this->cnt += required;
        if (this->cnt  > this->max_resources){
            printf("do not have so much resources");
            exit(0);

        }

        return  &this->resources[nn];
    }

};

#include<stdlib.h>
template <typename T>
class Resources_cpu
{
public:
    int max_resources;
    int cnt;
    T *resources;
public:

    Resources_cpu(int n){
        this->max_resources = n;
        this->cnt = 0;
        this->resources = (T*) malloc(sizeof(T) * this->max_resources);

    }


    T* allocate_resource(int required){
        int nn = this->cnt;
        this->cnt += required;
        if (this->cnt  >= this->max_resources){
            printf("do not have so much resources");
            exit(0);

        }

        return  &this->resources[nn];
    }

};
#endif // RESOURCES_H
