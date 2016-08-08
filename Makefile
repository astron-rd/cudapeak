PROGRAMS=cudapeak

default: ${PROGRAMS}

cudapeak: cudapeak.cu
	nvcc -o $@ $^ -lcuda -std=c++11

clean:
	@rm -f ${PROGRAMS}
