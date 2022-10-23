


cmake_minimum_required(VERSION 3.5)

# project name
project (SolverRosen)

 

add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++14  -mavx -mfma")

include_directories(.)
INCLUDE_DIRECTORIES("/usr/include/eigen3")

find_package(Eigen3 REQUIRED)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
	set (CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVERBOSE")

set(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)

INCLUDE_DIRECTORIES(./${CUDA_INCLUDE_DIRS})
include_directories(.)
set(CUDA_NVCC_FLAGS  --fmad=true -Xptxas -dlcm=ca -O3 )

set(CUDA_NVCC_FLAGS
    --relocatable-device-code=true -Xcompiler
    -fPIC)
LINK_DIRECTORIES("/usr/local/cuda-11.1/lib64")
INCLUDE_DIRECTORIES ("/usr/local/cuda-11.1/include")

INCLUDE_DIRECTORIES("/home/chenyidong/armadillo_lib/include")
LINK_DIRECTORIES("/home/chenyidong/armadillo_lib/lib")
LINK_DIRECTORIES("/home/chenyidong/hdf5_installed/lib")
INCLUDE_DIRECTORIES ("/home/chenyidong/hdf5_installed/include")

 

INCLUDE_DIRECTORIES (Cyclic)
add_subdirectory(Cyclic)
INCLUDE_DIRECTORIES (utils)
add_subdirectory(utils)



 
cuda_add_executable(Prog   test_prograd.cu  PCR_Class.cu  PCR_Device_Functions.cu
 cu_functors.cu  serial_tSolver.cu )
target_compile_options(Prog PUBLIC  )
target_include_directories(Prog PUBLIC )
target_link_libraries(Prog
        -lcublas
        -fopenmp
        -lcudart -lcusolver
        -larmadillo
        )
         



