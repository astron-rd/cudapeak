NVCC   := nvcc
CFLAGS := -lcuda --resource-usage -arch=native

SRC := $(wildcard *.cu)
SRC := $(filter-out common.cu, $(SRC))
BIN := $(patsubst %.cu, %.x, $(SRC))

default: ${BIN}

%.x: %.cu kernels/%.cu common.cu
	${NVCC} ${CFLAGS} -o $@ $^

clean:
	@rm -f ${BIN}
