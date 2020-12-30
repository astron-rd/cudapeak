PROGRAMS=dmem.x fp32.x fp32_smem.x fp32_dmem.x fp32_sincos_fpu.x fp32_sincos_sfu.x

.PRECIOUS: %.o

default: ${PROGRAMS}

%.x: %.cu kernels/%.o common.o
	nvcc -o $@ $^ -lcuda

%.o: %.cu
	nvcc -o $@ $^ -c -lineinfo

clean:
	@rm -f ${PROGRAMS} *.o kernels/*.o
