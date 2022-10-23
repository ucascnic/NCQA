# NCQA
computing the non-convex risk parity porfolio problems by the non-convex quadratic approxiamtion (NCQA),  interior point method (IPM) and sequence quadratic program (SQP)


## Environment Requirements

**Programming Language:** CUDA C/C++ (tested on cuda/11.1)

## Installation Instructions

For unified memory implementation

(1) In the `CMakeLists.txt`, edit the variable `CUDA_INSTALL_PATH` to match the CUDA installation directory.

(2) Type `cmake .` and  `make`  to compile.



## Reference

[1] G. Scutari, F. Facchinei, P. Song, D. P. Palomar, Decomposition by partial linearization: Parallel optimization of multi-agent systems, IEEE Transactions on Signal Processing 62 (3) (2014) 641–656.

[2] G. Scutari, F. Facchinei, L. Lampariello, Parallel and distributed methods for constrained nonconvex optimization part i: Theory, IEEE Transactions on Signal Processing 65 (8) (2017) 1929–1944. 

[3] M. Powell, Nonlinear programming–sequential unconstrained minimization techniques, The Computer Journal, 1990.

[4] C. T. Lawrence, A. L. Tits, A computationally efficient feasible sequential quadratic programming algorithm, SIAM Journal on Optimization 11 (4)(1998) 1092–1118.

[5] R. H. Byrd, J. Nocedal, R. A. Waltz, Feasible interior methods using slacks for nonlinear optimization, Computational Optimization and Applications 26 (1) (2003) 35–61.