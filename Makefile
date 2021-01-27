NVCC   := nvcc
CFLAGS := -lcuda --resource-usage --use_fast_math

SRC := $(wildcard *.cu)
SRC := $(filter-out common.cu, $(SRC))
BIN := $(patsubst %.cu, %.x, $(SRC))

default: ${BIN}

%.x: %.cu kernels/%.cu common.cu
	${NVCC} ${CFLAGS} -o $@ $^

clean:
	@rm -f ${BIN}
