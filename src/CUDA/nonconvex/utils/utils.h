#ifndef CU_UTILS_H
#define CU_UTILS_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/detail/vector_base.h>

#include <iostream>
#include <string>
#include <memory>
#include <functional>

namespace utils{

typedef thrust::host_vector<double> HostD;
typedef thrust::device_vector<double> DeviceD;

void PrintVector(bool toLogFile, std::string vector_name, HostD vector);
void PrintVector(std::string vector_name, HostD vector);
void PrintVector(std::string vector_name, DeviceD vector);

void LogProgramResults(std::string name,HostD vector);
void Log(std::string text);


}//END - namespace
#endif
