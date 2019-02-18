PROGRAMS=cudapeak.x
KERNELS=mem_global_kernels.o compute_sp_kernels.o compute_sp_sincos_fpu_kernels.o compute_sp_sincos_sfu_kernels.o compute_sp_ai_kernels.o

default: ${PROGRAMS}

cudapeak.x: cudapeak.cu ${KERNELS}
	nvcc -o $@ $^ -lcuda -std=c++11

%_kernels.o: %_kernels.cu
	nvcc -o $@ $^ -c

clean:
	@rm -f ${PROGRAMS} ${KERNELS}
