default:
	-rm -rf a.out
	nvcc -O2 -arch=sm_70 -lcublas main.cu
	./a.out
prof:
	-rm -rf a.out
	nvcc -g -G -arch=sm_70 -lcublas main.cu
	nsys profile --stats=true a.out