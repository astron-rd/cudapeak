PROGRAMS=cudapeak

default: ${PROGRAMS}

cudapeak: cudapeak.cu
	nvcc -o $@ $^ -lcuda -std=c++11 -g -lineinfo

clean:
	@rm -f ${PROGRAMS}
