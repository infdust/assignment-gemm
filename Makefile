default:
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -keep
	./a.out
unit_test:
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include test.cu -lgtest -lpthread
	./a.out
perf:
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=8192 -D MACRO_K=8192
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=8192 -D MACRO_K=2048
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=8192 -D MACRO_K=512
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=4096 -D MACRO_K=8192
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=4096 -D MACRO_K=2048
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=4096 -D MACRO_K=512
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=2048 -D MACRO_K=8192
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=2048 -D MACRO_K=2048
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=2048 -D MACRO_K=512
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=1024 -D MACRO_K=8192
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=1024 -D MACRO_K=2048
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=1024 -D MACRO_K=512
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=512 -D MACRO_K=8192
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=512 -D MACRO_K=2048
	./a.out
	-rm -rf a.out
	nvcc -O2 -arch=sm_80 -lcublas -I ../../cutlass/cutlass/include main.cu -D MACRO_MN=512 -D MACRO_K=512
	./a.out