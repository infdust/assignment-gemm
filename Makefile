default:
	-rm -rf gemm.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -o gemm.out -D COMPILING
	./gemm.out
prof:
	-rm -rf gemm.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -o gemm.out -lineinfo -D COMPILING
	-rm -rf profile.ncu-rep
	sudo /usr/local/cuda/bin/ncu -o profile --kernel-id :::"100" --set full ./gemm.out
unit_test:
	-rm -rf gemm.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include test.cu -o gemm.out -lgtest -lpthread -D COMPILING
	./gemm.out