cuda_add_library(Cyclic STATIC
   cu_cr_functors.cu  cu_cr_solver.cu
	)
target_link_libraries(Cyclic) 

