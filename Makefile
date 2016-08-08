PROGRAMS=cudapeak

default: ${PROGRAMS}

cudapeak: cudapeak.cu
	nvcc -o $@ $^ -lcuda

clean:
	@rm -f ${PROGRAMS}
