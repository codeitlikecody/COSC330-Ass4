# compiler, flags and libraries
COMPILER = nvcc
CFLAGS = -Xcompiler -Wall 

# all executables to compile
TARGET = mandelbrotCUDA
EXES = mandelbrotCUDA
OBJS = mandelbrotCUDA.o bmpfile.o

# main app to run
APPTORUN = mandelbrotCUDA
APPARGS = 1920 1080

# recipies
target: ${TARGET}

all: ${EXES}

mandelbrotCUDA: ${OBJS}
	${COMPILER} ${OBJS} -o mandelbrotCUDA ${CFLAGS}

mandelbrotCUDA.o: mandelbrotCUDA.cu
	${COMPILER} ${CFLAGS} mandelbrotCUDA.cu -c

%.o: %.c
	${COMPILER} ${CFLAGS} $< -c 

.PHONY: clean
clean:
	rm -f *~ *.o ${EXES}

.PHONY: run
run: ${APPTORUN}
	./${APPTORUN} ${APPARGS}

